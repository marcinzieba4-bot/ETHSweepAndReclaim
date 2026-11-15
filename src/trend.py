"""Daily trend following on Binance perps, with position-based costs and funding.

    python -m src.trend --symbols ETHUSDT BTCUSDT SOLUSDT --split 2025-01-01

Rules are evaluated on the daily close (00:00 UTC) and held for the next day.
  * TSMOM-L    : position = sign(L-day return), L in 7/14/28/56
  * DONCH-N/X  : enter long on a close above the N-day high, short on a close below the N-day low,
                 exit on a close through the opposite X-day channel (classic 20/10 and 55/20)
  * "_LO"      : long-only version (shorts become flat)
Costs: 0.07% per side x |position change|. Funding: a long pays, a short receives, the day's settled rates.
Optional vol targeting scales the position so the expected annual vol is `target_vol` (30-day realised vol).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIDE_COST = 0.0007


def daily(df5: pd.DataFrame) -> pd.DataFrame:
    d = df5.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    settle = df5["funding"][df5.index.hour.isin([0, 8, 16]) & (df5.index.minute == 0)]
    d["funding"] = settle.resample("1D").sum().reindex(d.index).fillna(0.0)   # paid by longs during the day
    return d.dropna(subset=["close"])


def tsmom(d: pd.DataFrame, L: int) -> pd.Series:
    return np.sign(np.log(d["close"] / d["close"].shift(L))).fillna(0.0)


def donchian(d: pd.DataFrame, n: int, x: int) -> pd.Series:
    c = d["close"].to_numpy()
    hi_n = d["high"].rolling(n).max().shift(1).to_numpy()
    lo_n = d["low"].rolling(n).min().shift(1).to_numpy()
    hi_x = d["high"].rolling(x).max().shift(1).to_numpy()
    lo_x = d["low"].rolling(x).min().shift(1).to_numpy()
    pos = np.zeros(len(c))
    p = 0.0
    for i in range(len(c)):
        if np.isnan(hi_n[i]):
            pos[i] = 0.0
            continue
        if p > 0 and c[i] < lo_x[i]:
            p = 0.0
        elif p < 0 and c[i] > hi_x[i]:
            p = 0.0
        if c[i] > hi_n[i]:
            p = 1.0
        elif c[i] < lo_n[i]:
            p = -1.0
        pos[i] = p
    return pd.Series(pos, index=d.index)


def backtest(d: pd.DataFrame, sig: pd.Series, target_vol: float | None = None) -> pd.DataFrame:
    ret = d["close"].pct_change().fillna(0.0)
    pos = sig.copy()
    if target_vol:
        rv = ret.rolling(30).std() * np.sqrt(365)
        pos = pos * (target_vol / rv).clip(upper=2.0).fillna(0.0)
    held = pos.shift(1).fillna(0.0)                     # signal at close t, held during day t+1
    turnover = held.diff().abs().fillna(held.abs())
    pnl = held * ret - turnover * SIDE_COST - held * d["funding"]
    return pd.DataFrame({"pnl": pnl, "pos": held, "turnover": turnover})


def metrics(bt: pd.DataFrame) -> dict:
    r = bt["pnl"]
    if len(r) < 30:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.0
    dd = 1 - eq / eq.cummax()
    vol = r.std() * np.sqrt(365)
    return dict(
        cagr=round(float(eq.iloc[-1] ** (1 / yrs) - 1), 3),
        vol=round(float(vol), 3),
        sharpe=round(float(r.mean() * 365 / vol), 2) if vol > 0 else np.nan,
        max_dd=round(float(dd.max()), 3),
        in_mkt=round(float((bt["pos"] != 0).mean()), 2),
        trades_yr=round(float((bt["turnover"] > 0).sum() / yrs), 1),
    )


def strategies(d: pd.DataFrame) -> dict:
    s = {f"TSMOM-{L}": tsmom(d, L) for L in (7, 14, 28, 56)}
    s["TSMOM-ENS"] = (s["TSMOM-14"] + s["TSMOM-28"] + s["TSMOM-56"]) / 3.0   # no single lookback to cherry-pick
    s["DONCH-20/10"] = donchian(d, 20, 10)
    s["DONCH-55/20"] = donchian(d, 55, 20)
    for k in list(s):
        s[k + "_LO"] = s[k].clip(lower=0.0)
    s["BUY&HOLD"] = pd.Series(1.0, index=d.index)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/trend")
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--target-vol", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(a.split, tz="UTC")
    rows = []
    for sym in a.symbols:
        d = daily(pd.read_parquet(Path(a.data) / f"{sym}_5m.parquet"))
        for name, sig in strategies(d).items():
            bt = backtest(d, sig, a.target_vol)
            bt.to_csv(out / f"{sym}_{name.replace('/', '-')}.csv")
            for part, m in (("IS", bt.index < split), ("OOS", bt.index >= split), ("ALL", bt.index == bt.index)):
                rows.append(dict(symbol=sym, strategy=name, part=part, **metrics(bt[m])))
    t = pd.DataFrame(rows)
    tag = f"_tv{a.target_vol}" if a.target_vol else ""
    t.to_csv(out / f"trend_summary{tag}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(t.pivot_table(index=["strategy"], columns=["symbol", "part"], values="sharpe", aggfunc="first").round(2).to_string())
        print()
        print(t.pivot_table(index=["strategy"], columns=["symbol", "part"], values="cagr", aggfunc="first").round(2).to_string())
        print()
        print(t.pivot_table(index=["strategy"], columns=["symbol", "part"], values="max_dd", aggfunc="first").round(2).to_string())


if __name__ == "__main__":
    main()
