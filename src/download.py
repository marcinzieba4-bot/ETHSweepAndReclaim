"""Download ETHUSDT USD-M perp data from data.binance.vision.

Datasets (5-minute resolution):
  * klines 5m          -> OHLCV + taker buy volume
  * metrics            -> sum_open_interest (coins), sum_open_interest_value (USD)
  * fundingRate        -> last_funding_rate per settlement

Usage:
    python -m src.download --start 2022-01 --end 2026-11 --out data
Output: data/ETHUSDT_5m.parquet (merged, UTC index)
"""
from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _get_zip_csv(url: str, cache: Path) -> pd.DataFrame | None:
    f = cache / url.rsplit("/", 1)[-1]
    if not f.exists():
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        f.write_bytes(r.content)
    with zipfile.ZipFile(f) as z:
        raw = z.read(z.namelist()[0])
    return pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)


def _months(start: str, end: str):
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    while s <= e:
        yield s
        s += 1


def _days(d0: date, d1: date):
    while d0 <= d1:
        yield d0
        d0 += timedelta(days=1)


def _strip_header(df: pd.DataFrame) -> pd.DataFrame:
    # newer files carry a header row; older ones don't
    if isinstance(df.iloc[0, 0], str) and not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    return df


def klines(sym, start, end, cache, workers=8):
    today = pd.Timestamp.utcnow().to_period("M")
    urls = []
    for m in _months(start, end):
        if m < today:
            urls.append(f"{BASE}/monthly/klines/{sym}/5m/{sym}-5m-{m}.zip")
        else:  # current month -> daily files
            for d in _days(m.start_time.date(), min(m.end_time.date(), date.today() - timedelta(days=1))):
                urls.append(f"{BASE}/daily/klines/{sym}/5m/{sym}-5m-{d}.zip")
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda u: _get_zip_csv(u, cache), urls) if p is not None]
    df = pd.concat([_strip_header(p) for p in parts], ignore_index=True)
    df.columns = KLINE_COLS
    df = df.astype({c: float for c in KLINE_COLS[1:6] + ["taker_buy_volume"]})
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df.set_index("ts")[["open", "high", "low", "close", "volume", "taker_buy_volume"]].sort_index()


def metrics(sym, start, end, cache, workers=16):
    d0 = max(pd.Period(start, "M").start_time.date(), date(2021, 12, 1))
    d1 = min(pd.Period(end, "M").end_time.date(), date.today() - timedelta(days=1))
    urls = [f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{d}.zip" for d in _days(d0, d1)]
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda u: _get_zip_csv(u, cache), urls) if p is not None]
    df = pd.concat([_strip_header(p) for p in parts], ignore_index=True)
    df = df.iloc[:, [0, 2, 3]]
    df.columns = ["create_time", "oi", "oi_usd"]
    df["ts"] = pd.to_datetime(df["create_time"], utc=True)
    return df.set_index("ts")[["oi", "oi_usd"]].astype(float).sort_index()


def funding(sym, start, end, cache):
    rows = []
    for m in _months(start, end):
        p = _get_zip_csv(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip", cache)
        if p is not None:
            rows.append(_strip_header(p))
    if not rows:
        return pd.DataFrame(columns=["funding"])
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df.iloc[:, 0].astype("int64"), unit="ms", utc=True)
    df["funding"] = df.iloc[:, 2].astype(float)
    return df.set_index("ts")[["funding"]].sort_index()


def build(sym="ETHUSDT", start="2022-01", end=None, out="data"):
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m")
    out = Path(out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    k = klines(sym, start, end, cache)
    m = metrics(sym, start, end, cache)
    f = funding(sym, start, end, cache)
    m.index = m.index.floor("5min")
    df = k.join(m[~m.index.duplicated()], how="left")
    df[["oi", "oi_usd"]] = df[["oi", "oi_usd"]].ffill()
    df["funding"] = f["funding"].reindex(df.index, method="ffill") if len(f) else 0.0
    df = df.dropna(subset=["oi"])
    path = out / f"{sym}_5m.parquet"
    df.to_parquet(path)
    print(f"saved {len(df):,} bars {df.index[0]} -> {df.index[-1]} to {path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    build(a.symbol, a.start, a.end, a.out)
