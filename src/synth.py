"""Synthetic ETH-like 5m data for pipeline tests only (NOT for conclusions)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def make(days=200, seed=1, p0=2000.0):
    rng = np.random.default_rng(seed)
    n = days * 288
    vol = np.empty(n)
    v = 0.002
    for i in range(n):  # GARCH-ish vol clustering
        v = np.sqrt(1e-7 + 0.05 * (v * rng.standard_normal()) ** 2 + 0.94 * v ** 2)
        vol[i] = v
    r = vol * rng.standard_t(4, n) / np.sqrt(2)
    c = p0 * np.exp(np.cumsum(r))
    o = np.r_[p0, c[:-1]]
    rng_hl = np.abs(rng.normal(0, vol, n)) * c
    h = np.maximum(o, c) + rng_hl
    l = np.minimum(o, c) - rng_hl
    volume = 5e4 * (1 + 50 * vol) * rng.lognormal(0, 0.3, n)
    buy = np.clip(0.5 + 30 * r + rng.normal(0, 0.05, n), 0.05, 0.95)
    doi = 0.0004 * rng.standard_normal(n) - 0.6 * np.abs(r) * (r < -2 * vol)
    oi = 2.5e6 * np.exp(np.cumsum(doi - doi.mean()))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=volume,
                           taker_buy_volume=volume * buy, oi=oi), index=idx)
    df["oi_usd"] = df["oi"] * df["close"]
    df["funding"] = 0.0001 * np.sign(pd.Series(r, index=idx).rolling(96, min_periods=1).sum())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--out", default="data/SYNTH_5m.parquet")
    a = ap.parse_args()
    d = make(a.days)
    d.to_parquet(a.out)
    print(f"saved {len(d):,} synthetic bars to {a.out}")
