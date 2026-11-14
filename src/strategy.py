"""Sweep & Reclaim on liquidation clusters.

LONG setup (SHORT is the mirror image):
  * candidate level L = strongest modelled long-liquidation cluster 1..6% below price,
    with size >= min_share of total OI. Refreshed when idle every `refresh` bars.
  * SWEEP : a bar's low trades below L (cluster liquidated).
  * fail  : price extends more than `max_ext` beyond L -> breakdown, setup cancelled.
  * RECLAIM: within `reclaim_bars` a bar CLOSES back above L*(1+buf)
             and OI fell by >= oi_drop during the sweep (leverage flushed)
             and funding <= funding_max (optional).
  * entry next bar open; stop below sweep extreme; target = strongest short cluster
    above (if >= tp_min_R), else fixed `tp_R`; time stop `max_hold` bars.

control=True replaces the cluster level with a level at a random distance drawn from
the empirical distance distribution of real clusters -> tells us whether the
liquidation map adds anything over "any level at a similar distance".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class StratConfig:
    side: str = "long"           # long | short | both
    min_share: float = 0.05      # cluster >= 5% of the modelled liquidation book
    refresh: int = 12            # bars between candidate refreshes when idle (1h)
    reclaim_bars: int = 12       # must reclaim within 1h of first sweep
    max_ext: float = 0.03        # sweep deeper than 3% beyond level = breakdown
    buf: float = 0.001           # reclaim close must be 0.1% back inside
    oi_drop: float = 0.005       # OI drop during sweep >= 0.5%
    oi_lookback: int = 12
    funding_max: float | None = None   # long only if funding <= this (short: >= -this)
    stop_buf: float = 0.001
    tp_R: float = 2.0
    tp_min_R: float = 1.0
    use_map_tp: bool = True
    max_hold: int = 288          # 24h
    fee: float = 0.0005          # taker per side
    slip: float = 0.0002         # per side
    control: bool = False
    seed: int = 0


def _one_side(df, mp, cfg: StratConfig, sgn: int, rng):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    oi = df["oi"].to_numpy(float)
    fund = df["funding"].to_numpy(float) if "funding" in df else np.zeros(len(df))
    if sgn > 0:
        lvl, share, tlvl = mp["L_lvl"].to_numpy(), mp["L_share"].to_numpy(), mp["S_lvl"].to_numpy()
    else:
        lvl, share, tlvl = mp["S_lvl"].to_numpy(), mp["S_share"].to_numpy(), mp["L_lvl"].to_numpy()
    dist_pool = np.abs(lvl / c - 1)[(share >= cfg.min_share) & np.isfinite(lvl)]
    if cfg.control and dist_pool.size == 0:
        return []

    n = len(df)
    trades = []
    state, L, since, ext, oi_ref, last_ref = "idle", np.nan, 0, np.nan, np.nan, -10**9
    t = 0
    while t < n - 1:
        # --- refresh candidate
        if state == "idle" and t - last_ref >= cfg.refresh:
            last_ref = t
            if cfg.control:
                d = rng.choice(dist_pool)
                L = c[t] * (1 - sgn * d)
            else:
                L = lvl[t] if (share[t] >= cfg.min_share and np.isfinite(lvl[t])) else np.nan
            t += 1
            continue
        if state == "idle":
            if np.isfinite(L) and (l[t] < L if sgn > 0 else h[t] > L):
                state, since = "swept", t
                ext = l[t] if sgn > 0 else h[t]
                oi_ref = oi[max(0, t - cfg.oi_lookback):t + 1].max()
            t += 1
            continue
        # --- swept: wait for reclaim
        ext = min(ext, l[t]) if sgn > 0 else max(ext, h[t])
        if sgn * (L - ext) / L > cfg.max_ext or t - since > cfg.reclaim_bars:
            state, L, last_ref = "idle", np.nan, -10**9
            t += 1
            continue
        reclaimed = (c[t] > L * (1 + cfg.buf)) if sgn > 0 else (c[t] < L * (1 - cfg.buf))
        if not reclaimed:
            t += 1
            continue
        oi_ok = (oi_ref - oi[since:t + 1].min()) / oi_ref >= cfg.oi_drop
        f_ok = cfg.funding_max is None or (fund[t] * sgn <= cfg.funding_max)
        if not (oi_ok and f_ok):
            state, L, last_ref = "idle", np.nan, -10**9
            t += 1
            continue
        # --- entry at next open
        e_i = t + 1
        entry = o[e_i] * (1 + sgn * cfg.slip)
        stop = ext * (1 - sgn * cfg.stop_buf)
        risk = sgn * (entry - stop)
        if risk <= 0:
            state, L, last_ref = "idle", np.nan, -10**9
            t += 1
            continue
        tp = entry + sgn * cfg.tp_R * risk
        tp_src = "R"
        if cfg.use_map_tp and np.isfinite(tlvl[t]):
            r_map = sgn * (tlvl[t] - entry) / risk
            if r_map >= cfg.tp_min_R:
                tp, tp_src = tlvl[t], "map"
        exit_p, why, j = None, "time", e_i
        for j in range(e_i, min(n, e_i + cfg.max_hold)):
            hit_stop = l[j] <= stop if sgn > 0 else h[j] >= stop
            hit_tp = h[j] >= tp if sgn > 0 else l[j] <= tp
            if hit_stop:  # conservative: stop first if both in one bar
                exit_p = (min(o[j], stop) if sgn > 0 else max(o[j], stop)) if j > e_i else stop
                why = "stop"
                break
            if hit_tp:
                exit_p, why = tp, "tp"
                break
        if exit_p is None:
            exit_p = c[j]
        exit_p *= (1 - sgn * cfg.slip)
        gross = sgn * (exit_p - entry) / entry
        net = gross - 2 * cfg.fee
        trades.append(dict(
            side="long" if sgn > 0 else "short", signal_ts=df.index[t], entry_ts=df.index[e_i],
            exit_ts=df.index[j], level=L, entry=entry, stop=stop, tp=tp, tp_src=tp_src,
            exit=exit_p, why=why, ret=net, R=(net * entry) / risk, risk_pct=risk / entry,
            bars=j - e_i + 1, weekend=df.index[e_i].dayofweek >= 5,
        ))
        state, L, last_ref = "idle", np.nan, -10**9
        t = j + 1
    return trades


def run(df: pd.DataFrame, mp: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    sides = {"long": [1], "short": [-1], "both": [1, -1]}[cfg.side]
    tr = []
    for s in sides:
        tr += _one_side(df, mp, cfg, s, rng)
    out = pd.DataFrame(tr)
    if len(out):
        out = out.sort_values("entry_ts").reset_index(drop=True)
    return out


def stats(tr: pd.DataFrame, risk_per_trade: float = 0.01) -> dict:
    if tr is None or len(tr) == 0:
        return dict(n=0)
    R = tr["R"].to_numpy()
    eq = np.cumprod(1 + risk_per_trade * R)
    dd = 1 - eq / np.maximum.accumulate(eq)
    wins, losses = R[R > 0].sum(), -R[R < 0].sum()
    yrs = max((tr["exit_ts"].iloc[-1] - tr["entry_ts"].iloc[0]).days / 365.25, 1e-9)
    return dict(
        n=int(len(tr)), per_year=round(len(tr) / yrs, 1),
        hit=round(float((R > 0).mean()), 3), avg_R=round(float(R.mean()), 3),
        med_R=round(float(np.median(R)), 3), pf=round(float(wins / losses), 2) if losses > 0 else np.inf,
        t_stat=round(float(R.mean() / (R.std(ddof=1) / np.sqrt(len(R)))), 2) if len(R) > 2 else np.nan,
        tot_ret_1pct=round(float(eq[-1] - 1), 3), max_dd_1pct=round(float(dd.max()), 3),
        avg_hold_h=round(float(tr["bars"].mean() * 5 / 60), 1),
        tp_map_share=round(float((tr["tp_src"] == "map").mean()), 2),
    )


def cfg_dict(cfg):
    return asdict(cfg)
