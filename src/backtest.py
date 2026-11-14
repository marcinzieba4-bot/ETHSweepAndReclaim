"""End-to-end backtest: liquidation map -> sweep&reclaim -> robustness report.

    python -m src.backtest --data data/ETHUSDT_5m.parquet --out results --split 2025-01-01

Steps
  1. build (or load cached) liquidation map
  2. baseline config, long / short / both, full sample
  3. small parameter grid on IN-SAMPLE only, pick best by t-stat (min trades)
  4. evaluate picked config OUT-OF-SAMPLE
  5. control: same rules on random levels at matched distances (N seeds)
  6. breakdowns: weekend vs weekday, by year, by exit reason
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .liqmap import MapConfig, build_map
from .strategy import StratConfig, run, stats, cfg_dict

GRID = dict(
    min_share=[0.03, 0.05, 0.10],
    reclaim_bars=[6, 12, 24],
    oi_drop=[0.0, 0.005, 0.01],
    tp_R=[1.5, 2.0, 3.0],
)


def load_map(df, cache: Path, mcfg: MapConfig):
    key = f"map_hl{mcfg.half_life_days}_b{mcfg.band_min}-{mcfg.band_max}_{df.index[0]:%Y%m%d}_{df.index[-1]:%Y%m%d%H%M}.parquet"
    f = cache / key
    if f.exists():
        return pd.read_parquet(f)
    mp = build_map(df, mcfg)
    mp.to_parquet(f)
    return mp


def by(tr, col):
    return {str(k): stats(g) for k, g in tr.groupby(col)} if len(tr) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ETHUSDT_5m.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--min-trades", type=int, default=30)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.data)
    mcfg = MapConfig()
    mp = load_map(df, out, mcfg)
    split = pd.Timestamp(a.split, tz="UTC")
    is_df, oos_df = df[df.index < split], df[df.index >= split]
    is_mp, oos_mp = mp.loc[is_df.index], mp.loc[oos_df.index]
    rep = dict(data=dict(bars=len(df), start=str(df.index[0]), end=str(df.index[-1]), split=str(split)),
               map_cfg=mcfg.__dict__)

    # 2) baseline
    base = StratConfig()
    rep["baseline"] = {}
    for side in ("long", "short", "both"):
        tr = run(df, mp, replace(base, side=side))
        rep["baseline"][side] = stats(tr)
        if side == "both":
            tr.to_csv(out / "trades_baseline.csv", index=False)
            base_tr = tr

    # 3) grid on IS
    rows = []
    keys = list(GRID)
    for side in ("long", "short"):
        for vals in itertools.product(*GRID.values()):
            cfg = replace(base, side=side, **dict(zip(keys, vals)))
            s = stats(run(is_df, is_mp, cfg))
            rows.append({**dict(zip(keys, vals)), "side": side, **s})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "grid_is.csv", index=False)
    rep["grid_size"] = len(grid)
    rep["grid_share_positive_avgR"] = round(float((grid["avg_R"] > 0).mean()), 3) if "avg_R" in grid else None

    picked = {}
    for side in ("long", "short"):
        g = grid[(grid.side == side) & (grid.n >= a.min_trades)].dropna(subset=["t_stat"])
        if g.empty:
            continue
        best = g.sort_values("t_stat", ascending=False).iloc[0]
        cfg = replace(base, side=side, **{k: best[k].item() if hasattr(best[k], "item") else best[k] for k in keys})
        cfg.reclaim_bars = int(cfg.reclaim_bars)
        tr_oos = run(oos_df, oos_mp, cfg)
        tr_oos.to_csv(out / f"trades_oos_{side}.csv", index=False)
        # 5) control on OOS with identical rules
        ctrl = [stats(run(oos_df, oos_mp, replace(cfg, control=True, seed=s))) for s in range(a.seeds)]
        ctrl_avg = np.array([c.get("avg_R", np.nan) for c in ctrl], float)
        real = stats(tr_oos)
        picked[side] = dict(
            params={k: getattr(cfg, k) for k in keys},
            is_stats={k: float(best[k]) for k in ("n", "hit", "avg_R", "pf", "t_stat")},
            oos_stats=real,
            control_oos=dict(seeds=a.seeds, avg_R_mean=round(float(np.nanmean(ctrl_avg)), 3),
                             avg_R_p90=round(float(np.nanpercentile(ctrl_avg, 90)), 3),
                             real_beats_share=round(float(np.nanmean(real.get("avg_R", -9) > ctrl_avg)), 3)),
            oos_weekend=by(tr_oos, "weekend"),
            oos_exit=by(tr_oos, "why"),
        )
    rep["picked"] = picked

    # 6) breakdowns on baseline
    if len(base_tr):
        base_tr["year"] = pd.to_datetime(base_tr["entry_ts"]).dt.year
        rep["baseline_by_year"] = by(base_tr, "year")
        rep["baseline_weekend"] = by(base_tr, "weekend")
        rep["baseline_exit"] = by(base_tr, "why")
        rep["baseline_control_full"] = {}
        for side in ("long", "short"):
            c = [stats(run(df, mp, replace(base, side=side, control=True, seed=s))).get("avg_R", np.nan)
                 for s in range(min(a.seeds, 10))]
            rep["baseline_control_full"][side] = round(float(np.nanmean(c)), 3)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            for side, g in base_tr.groupby("side"):
                ax.plot(pd.to_datetime(g["exit_ts"]), np.cumsum(g["R"]), label=f"{side} (cum R)")
            ax.axvline(split, ls="--", c="grey")
            ax.set_title("Sweep & Reclaim baseline — cumulative R")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "equity_baseline.png", dpi=120)
        except Exception as e:  # plotting is optional
            print("plot skipped:", e)

    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
