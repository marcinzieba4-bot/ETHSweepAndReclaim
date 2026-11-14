"""Robustness checks that map the original trade plan onto the backtest.

Run after src.backtest (it reuses results/report.json and the cached map):

    python -m src.robustness --data data/ETHUSDT_5m.parquet --out results --split 2025-01-01

For the README-default config and for the IS-picked config of each side, it runs these variants:
  * as is
  * funding <= 0                (plan: "funding drops to zero or below" for the long)
  * funding below neutral 0.01% (milder version of the same filter)
  * reclaim confirmed on a 15m close, and on a 1h close (plan: "holds on a 15m or 1h close")
  * zero fees and slippage      (is there a gross edge that costs remove?)
Each variant gets IS stats, OOS stats and an OOS random-level control.
Output: results/robustness.json and results/robustness.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .liqmap import MapConfig
from .backtest import load_map
from .strategy import StratConfig, run, stats

G = {}


def _job(args):
    name, cfg, part, seed = args
    df, mp = G[part]
    c = cfg if seed is None else replace(cfg, control=True, seed=seed)
    tr = run(df, mp, c)
    s = stats(tr)
    if len(tr):
        s["avg_risk_pct"] = round(float(tr["risk_pct"].mean() * 100), 3)
        s["cost_R"] = round(float((2 * (c.fee + c.slip) / tr["risk_pct"]).mean()), 3)
        wk = tr[tr["weekend"]]
        wd = tr[~tr["weekend"]]
        s["weekend_avg_R"] = round(float(wk["R"].mean()), 3) if len(wk) else None
        s["weekday_avg_R"] = round(float(wd["R"].mean()), 3) if len(wd) else None
        s["n_weekend"] = int(len(wk))
    return name, part, seed, s


def variants(cfg: StratConfig):
    rb = cfg.reclaim_bars
    return {
        "as_is": cfg,
        "funding<=0": replace(cfg, funding_max=0.0),
        "funding<neutral": replace(cfg, funding_max=0.0000999),
        "confirm_15m": replace(cfg, confirm_min=15, reclaim_bars=max(rb, 12)),
        "confirm_1h": replace(cfg, confirm_min=60, reclaim_bars=max(rb, 24)),
        "zero_cost": replace(cfg, fee=0.0, slip=0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ETHUSDT_5m.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    df = pd.read_parquet(a.data)
    mp = load_map(df, out, MapConfig())
    split = pd.Timestamp(a.split, tz="UTC")
    is_df, oos_df = df[df.index < split], df[df.index >= split]
    G["is"] = (is_df, mp.loc[is_df.index])
    G["oos"] = (oos_df, mp.loc[oos_df.index])
    rep = json.loads((out / "report.json").read_text())

    base = StratConfig()
    configs = {}
    for side in ("long", "short"):
        configs[f"default_{side}"] = replace(base, side=side)
        if side in rep.get("picked", {}):
            p = rep["picked"][side]["params"]
            configs[f"picked_{side}"] = replace(base, side=side, **{**p, "reclaim_bars": int(p["reclaim_bars"])})

    jobs = []
    for cname, cfg in configs.items():
        for vname, vcfg in variants(cfg).items():
            name = f"{cname}|{vname}"
            jobs += [(name, vcfg, "is", None), (name, vcfg, "oos", None)]
            jobs += [(name, vcfg, "oos", s) for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        res = pool.map(_job, jobs, chunksize=4)

    table = {}
    for name, part, seed, s in res:
        row = table.setdefault(name, {"ctrl": []})
        if seed is None:
            row[part] = s
        else:
            row["ctrl"].append(s.get("avg_R", np.nan))
    rows = []
    for name, r in table.items():
        ctrl = np.array(r["ctrl"], float)
        real = r["oos"].get("avg_R", np.nan)
        cfg_name, var = name.split("|")
        rows.append(dict(
            config=cfg_name, variant=var,
            is_n=r["is"].get("n", 0), is_hit=r["is"].get("hit"), is_avg_R=r["is"].get("avg_R"), is_t=r["is"].get("t_stat"),
            oos_n=r["oos"].get("n", 0), oos_hit=r["oos"].get("hit"), oos_avg_R=real, oos_t=r["oos"].get("t_stat"),
            oos_pf=r["oos"].get("pf"), oos_ret_1pct=r["oos"].get("tot_ret_1pct"), oos_dd_1pct=r["oos"].get("max_dd_1pct"),
            oos_weekday_R=r["oos"].get("weekday_avg_R"), oos_weekend_R=r["oos"].get("weekend_avg_R"),
            oos_n_weekend=r["oos"].get("n_weekend"),
            oos_risk_pct=r["oos"].get("avg_risk_pct"), oos_cost_R=r["oos"].get("cost_R"),
            ctrl_avg_R=round(float(np.nanmean(ctrl)), 3) if np.isfinite(ctrl).any() else None,
            real_beats_ctrl=round(float(np.nanmean(real > ctrl)), 2) if np.isfinite(real) and np.isfinite(ctrl).any() else None,
        ))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "robustness.csv", index=False)
    (out / "robustness.json").write_text(json.dumps(dict(seeds=a.seeds, rows=rows), indent=2, default=str))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
