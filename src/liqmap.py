"""Model-based liquidation map (Coinglass-style), rebuilt bar by bar with no look-ahead.

Mechanics per 5m bar:
  1. Positions resting in the book whose liquidation price was touched this bar are
     liquidated (long bins >= bar low, short bins <= bar high) and removed.
  2. Voluntary closes (OI drop not explained by modelled liquidations) shrink both
     books proportionally.
  3. New OI (dOI > 0) is opened at the bar's typical price, split long/short by the
     taker-buy share, and spread across leverage cohorts -> liquidation prices
        long  = p * (1 - 1/L + mmr),   short = p * (1 + 1/L - mmr)
  4. Everything decays with a half-life (positions closed in ways we can't observe).

After the bar is processed a snapshot of the strongest long cluster below price and
the strongest short cluster above price (within a distance band) is stored. Those
snapshots are what the strategy sees at the bar close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MapConfig:
    lev: dict = field(default_factory=lambda: {5: 0.10, 10: 0.30, 25: 0.30, 50: 0.20, 100: 0.10})
    mmr: float = 0.004            # maintenance margin
    bin_step: float = 0.0025      # log-price bin width (0.25%)
    p_min: float = 300.0
    p_max: float = 10000.0
    half_life_days: float = 7.0
    cluster_bins: int = 3         # cluster = sum of N adjacent bins (~0.75%)
    band_min: float = 0.01        # look for clusters 1%..
    band_max: float = 0.06        # ..6% away from price
    bar_minutes: int = 5


def build_map(df: pd.DataFrame, cfg: MapConfig = MapConfig()) -> pd.DataFrame:
    edges = np.exp(np.arange(np.log(cfg.p_min), np.log(cfg.p_max), cfg.bin_step))
    nb = len(edges)
    longs = np.zeros(nb)   # coins resting with liq price in bin
    shorts = np.zeros(nb)
    decay = 0.5 ** (cfg.bar_minutes / (cfg.half_life_days * 1440))
    levs = np.array(list(cfg.lev.keys()), float)
    w = np.array(list(cfg.lev.values()), float)
    w = w / w.sum()
    k = np.ones(cfg.cluster_bins)

    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    vol = df["volume"].to_numpy(float)
    tb = df["taker_buy_volume"].to_numpy(float)
    oi = df["oi"].to_numpy(float)
    n = len(df)
    out = {x: np.full(n, np.nan) for x in
           ("L_lvl", "L_amt", "S_lvl", "S_amt", "liq_long", "liq_short", "book_long", "book_short")}

    def idx(p):
        return np.clip(np.searchsorted(edges, p) - 1, 0, nb - 1)

    def strongest(book, lo_p, hi_p):
        i0, i1 = idx(lo_p), idx(hi_p) + 1
        seg = book[i0:i1]
        if seg.size < cfg.cluster_bins or seg.max() <= 0:
            return np.nan, 0.0
        sm = np.convolve(seg, k, mode="same")
        j = int(sm.argmax())
        return edges[i0 + j], sm[j] * edges[i0 + j]   # level, USD notional

    prev_oi = oi[0]
    for t in range(n):
        longs *= decay
        shorts *= decay
        # 1) liquidations
        jl = idx(l[t])
        liq_l = longs[jl:].sum()
        longs[jl:] = 0.0
        js = idx(h[t]) + 1
        liq_s = shorts[:js].sum()
        shorts[:js] = 0.0
        # 2) OI change
        d = oi[t] - prev_oi
        prev_oi = oi[t]
        if d < 0:
            vol_close = max(0.0, -d - liq_l - liq_s)
            tot = longs.sum() + shorts.sum()
            if tot > 0:
                f = max(0.0, 1.0 - vol_close / tot)
                longs *= f
                shorts *= f
        elif d > 0:
            p = (h[t] + l[t] + c[t]) / 3.0
            buy = np.clip(tb[t] / vol[t], 0.3, 0.7) if vol[t] > 0 else 0.5
            np.add.at(longs, idx(p * (1 - 1 / levs + cfg.mmr)), d * buy * w)
            np.add.at(shorts, idx(p * (1 + 1 / levs - cfg.mmr)), d * (1 - buy) * w)
        # 3) snapshot at close
        px = c[t]
        out["L_lvl"][t], out["L_amt"][t] = strongest(longs, px * (1 - cfg.band_max), px * (1 - cfg.band_min))
        out["S_lvl"][t], out["S_amt"][t] = strongest(shorts, px * (1 + cfg.band_min), px * (1 + cfg.band_max))
        out["liq_long"][t] = liq_l * l[t]
        out["liq_short"][t] = liq_s * h[t]
        out["book_long"][t] = (longs * edges).sum()
        out["book_short"][t] = (shorts * edges).sum()

    res = pd.DataFrame(out, index=df.index)
    book = (res["book_long"] + res["book_short"]).replace(0, np.nan)
    res["L_share"] = res["L_amt"] / book      # cluster size relative to the modelled book
    res["S_share"] = res["S_amt"] / book
    return res
