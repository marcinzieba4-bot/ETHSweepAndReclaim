"""Event studies: what happens after liquidation / positioning events, at fixed horizons.

No stops or targets: entry at the next bar open, exit H bars later at the open, minus a
round-trip cost. That keeps the free parameters to the event definition only.

    python -m src.research --symbols ETHUSDT BTCUSDT SOLUSDT --split 2025-01-01

Hypotheses (direction stated before testing, return reported in that direction):
  H1 long_flush     : 1h move <= -k sigma and OI 1h change <= -m   -> long  (forced selling reverts)
  H2 short_squeeze  : 1h move >= +k sigma and OI 1h change <= -m   -> short (forced buying reverts)
  H3 new_shorts     : 1h move <= -k sigma and OI 1h change >= +m   -> short (fresh positioning continues)
  H4 new_longs      : 1h move >= +k sigma and OI 1h change >= +m   -> long
  H5 funding_high   : funding at settlement >= IS p95              -> short (crowded longs unwind)
     funding_low    : funding at settlement <= IS p5               -> long
  ref trend_20d     : at 00:00 UTC, sign of 20-day return          -> that direction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.0014                      # round trip: 2 x (0.05% taker + 0.02% slippage)
HORIZONS = {"1h": 12, "4h": 48, "12h": 144, "24h": 288}


def features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    lc = np.log(df["close"])
    r5 = lc.diff()
    f["sig1h"] = r5.rolling(288 * 3, min_periods=288).std() * np.sqrt(12)   # 3-day vol, 1h scale
    f["r1h"] = lc - lc.shift(12)
    f["z1h"] = f["r1h"] / f["sig1h"]
    f["oi1h"] = np.log(df["oi"] / df["oi"].shift(12))
    f["funding"] = df["funding"]
    f["r20d"] = lc - lc.shift(288 * 20)
    f["minute_of_day"] = df.index.hour * 60 + df.index.minute
    return f


def fwd_returns(df: pd.DataFrame) -> dict:
    o = df["open"].to_numpy(float)
    n = len(o)
    out = {}
    for name, h in HORIZONS.items():
        r = np.full(n, np.nan)
        i = np.arange(n - 1 - h)
        r[i] = o[i + 1 + h] / o[i + 1] - 1.0
        out[name] = r
    return out


def non_overlapping(idx: np.ndarray, gap: int) -> np.ndarray:
    keep, last = [], -10**12
    for i in idx:
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def event_stats(sig: np.ndarray, side: int, fwd: dict, index: pd.DatetimeIndex) -> dict:
    """sig: bool array of events at bar close. Returns stats per horizon, net of cost."""
    res = {}
    ev = np.flatnonzero(sig)
    for name, h in HORIZONS.items():
        e = non_overlapping(ev, h)
        r = side * fwd[name][e] - COST
        r = r[np.isfinite(r)]
        e = e[: len(r)]
        if len(r) < 3:
            res[name] = dict(n=int(len(r)))
            continue
        yrs = pd.Series(r, index=index[e]).groupby(index[e].year).mean()
        res[name] = dict(
            n=int(len(r)), mean_bp=round(float(r.mean() * 1e4), 1), hit=round(float((r > 0).mean()), 3),
            t=round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2),
            years_pos=f"{int((yrs > 0).sum())}/{len(yrs)}",
            by_year={int(k): round(float(v * 1e4), 1) for k, v in yrs.items()},
        )
    return res


def signals(f: pd.DataFrame, k: float, m: float, fund_hi: float, fund_lo: float) -> dict:
    z, oi = f["z1h"].to_numpy(), f["oi1h"].to_numpy()
    settle = np.isin(f["minute_of_day"].to_numpy(), [0, 480, 960])   # bar opening at a funding settlement
    midnight = f["minute_of_day"].to_numpy() == 0
    fund, r20 = f["funding"].to_numpy(), f["r20d"].to_numpy()
    with np.errstate(invalid="ignore"):
        return {
            "H1_long_flush": ((z <= -k) & (oi <= -m), +1),
            "H2_short_squeeze": ((z >= k) & (oi <= -m), -1),
            "H3_new_shorts": ((z <= -k) & (oi >= m), -1),
            "H4_new_longs": ((z >= k) & (oi >= m), +1),
            "H5_funding_high": (settle & (fund >= fund_hi), -1),
            "H5_funding_low": (settle & (fund <= fund_lo), +1),
            "ref_trend_long": (midnight & (r20 > 0), +1),
            "ref_trend_short": (midnight & (r20 < 0), -1),
            "ref_all_long": (midnight, +1),
        }


def run_symbol(sym: str, data_dir: Path, split: pd.Timestamp, k: float, m: float):
    df = pd.read_parquet(data_dir / f"{sym}_5m.parquet")
    f = features(df)
    fwd = fwd_returns(df)
    is_mask = df.index < split
    fund_is = f.loc[is_mask, "funding"]
    fund_hi, fund_lo = fund_is.quantile(0.95), fund_is.quantile(0.05)
    out = {}
    for name, (sig, side) in signals(f, k, m, fund_hi, fund_lo).items():
        sig = np.nan_to_num(sig, nan=0).astype(bool)
        out[name] = {
            "IS": event_stats(sig & is_mask, side, fwd, df.index),
            "OOS": event_stats(sig & ~is_mask, side, fwd, df.index),
        }
    return out, dict(fund_hi=fund_hi, fund_lo=fund_lo, end=str(df.index[-1]))


def table(res: dict, horizon: str) -> pd.DataFrame:
    rows = []
    for sym, r in res.items():
        for name, parts in r.items():
            for part, s in parts.items():
                x = s[horizon]
                rows.append(dict(symbol=sym, event=name, part=part, n=x.get("n"), mean_bp=x.get("mean_bp"),
                                 hit=x.get("hit"), t=x.get("t"), years_pos=x.get("years_pos")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ETHUSDT"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/research")
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--m", type=float, default=0.02)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(a.split, tz="UTC")
    res, meta = {}, {}
    for s in a.symbols:
        res[s], meta[s] = run_symbol(s, Path(a.data), split, a.k, a.m)
    (out / f"events_k{a.k}_m{a.m}.json").write_text(json.dumps(dict(meta=meta, res=res), indent=1, default=str))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        for h in HORIZONS:
            t = table(res, h)
            t.to_csv(out / f"events_k{a.k}_m{a.m}_{h}.csv", index=False)
            print(f"\n===== horizon {h}  (k={a.k} sigma, m={a.m:.0%} OI) =====")
            print(t.pivot_table(index=["event", "symbol"], columns="part",
                                values=["n", "mean_bp", "t", "years_pos"], aggfunc="first").to_string())


if __name__ == "__main__":
    main()
