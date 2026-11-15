"""Multi-coin trend portfolio: backtest, chart and today's positions.

    python -m src.download --symbol ETHUSDT && python -m src.download --symbol BTCUSDT && python -m src.download --symbol SOLUSDT
    python -m src.trend_signal --target-vol 0.3

Per coin: signal = mean of sign(14d, 28d, 56d return) in {-1, -1/3, +1/3, +1}, taken at the 00:00 UTC
daily close. Size = signal x target_vol / 30-day realised vol (capped at 2x). The portfolio is an
equal-weight average of the coins. Costs 0.07% per side on position changes, perp funding paid/received.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .trend import backtest, daily, metrics, tsmom

LOOKBACKS = (14, 28, 56)


def ensemble(d: pd.DataFrame) -> pd.Series:
    return sum(tsmom(d, L) for L in LOOKBACKS) / len(LOOKBACKS)


def chart(curves: dict, split: pd.Timestamp, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"Trend long/short": "#2a78d6", "Trend long-only": "#eb6834", "Buy & hold (same vol)": "#8f8d86"}
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, lw=2, color=colors[name], label=name, solid_capstyle="round")
        ax.annotate(f"{name}  {eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=ink)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.axvline(split, color=ink2, lw=1, ls=(0, (3, 3)))
    ax.text(split, ax.get_ylim()[1], "  out-of-sample →", va="top", fontsize=8.5, color=ink2)
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5, length=0)
    ax.set_ylabel("Growth of $1 (log scale)", color=ink2, fontsize=9)
    ax.set_title("ETH + BTC + SOL trend portfolio vs buy & hold, 30% vol target, after fees and funding",
                 loc="left", fontsize=10.5, color=ink)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=ink)
    ax.set_xlim(right=ax.get_xlim()[1] + 160)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT", "SOLUSDT"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/trend")
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--target-vol", type=float, default=0.3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(a.split, tz="UTC")

    D = {s: daily(pd.read_parquet(Path(a.data) / f"{s}_5m.parquet")) for s in a.symbols}
    streams = {"Trend long/short": {}, "Trend long-only": {}, "Buy & hold (same vol)": {}}
    live = {}
    for s, d in D.items():
        e = ensemble(d)
        streams["Trend long/short"][s] = backtest(d, e, a.target_vol)["pnl"]
        streams["Trend long-only"][s] = backtest(d, e.clip(lower=0), a.target_vol)["pnl"]
        streams["Buy & hold (same vol)"][s] = backtest(d, pd.Series(1.0, index=d.index), a.target_vol)["pnl"]
        c = d["close"]
        rv = float(c.pct_change().rolling(30).std().iloc[-1] * np.sqrt(365))
        lev = min(a.target_vol / rv, 2.0)
        live[s] = dict(
            as_of=str(c.index[-1] + pd.Timedelta(days=1)), close=float(c.iloc[-1]),
            **{f"ret_{L}d": round(float(c.iloc[-1] / c.iloc[-1 - L] - 1), 4) for L in LOOKBACKS},
            signal=round(float(e.iloc[-1]), 2), vol_30d=round(rv, 3),
            size_long_short=round(float(e.iloc[-1] * lev), 2), size_long_only=round(float(max(e.iloc[-1], 0) * lev), 2),
            # tomorrow's L-day return compares tomorrow's close with the close L-1 days before today's
            flip_levels={f"{L}d": float(c.iloc[-L]) for L in LOOKBACKS},
        )

    rows, curves = [], {}
    for name, per_coin in streams.items():
        r = pd.concat(per_coin, axis=1).mean(axis=1)
        curves[name] = (1 + r).cumprod()
        bt = pd.DataFrame({"pnl": r, "pos": 1.0, "turnover": 0.0})
        m_all, m_is, m_oos = metrics(bt), metrics(bt[bt.index < split]), metrics(bt[bt.index >= split])
        yearly = ((1 + r).groupby(r.index.year).prod() - 1).round(3)
        rows.append(dict(portfolio=name, sharpe=m_all["sharpe"], sharpe_IS=m_is["sharpe"], sharpe_OOS=m_oos["sharpe"],
                         cagr=m_all["cagr"], vol=m_all["vol"], max_dd=m_all["max_dd"],
                         **{str(y): v for y, v in yearly.items()}))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "portfolio_equity.csv")
    (out / "live_signal.json").write_text(json.dumps(live, indent=2))
    chart(curves, split, out / "portfolio_equity.png")
    with pd.option_context("display.width", 220):
        print(summary.to_string(index=False))
    print(f"\nLive positions (vol target {a.target_vol:.0%}), x equity per coin, portfolio holds 1/{len(D)} of each:")
    for s, v in live.items():
        flips = ", ".join(f"{k} above {p:,.2f}" for k, p in v["flip_levels"].items())
        print(f"  {s} for {v['as_of'][:10]}: signal {v['signal']:+.2f} | long/short {v['size_long_short']:+.2f}x | "
              f"long-only {v['size_long_only']:+.2f}x | flips: {flips}")


if __name__ == "__main__":
    main()
