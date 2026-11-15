# ETH Sweep & Reclaim

> **Result (2022-01 → 2026-11, real Binance data): no edge.** About −0.25R per trade after costs, negative every year,
> and no better than random levels. See [RESULTS.md](RESULTS.md).
>
> **What may work instead:** a 14/28/56-day trend ensemble on ETH, BTC and SOL, sized by volatility.
> Portfolio Sharpe 0.88 long/short (1.08 long-only), positive in- and out-of-sample on every coin. See [TREND.md](TREND.md).

Backtest of a liquidation-cluster **sweep & reclaim** setup on ETHUSDT perpetual (Binance USD-M, 5m bars).

## Hypothesis
When price runs through a dense cluster of modelled long liquidations (sweep), leverage is flushed.
If price then closes back above that level quickly (reclaim) while open interest has dropped,
the move was forced selling rather than real supply, so a mean-reversion long towards the nearest
short-liquidation cluster has positive expectancy. The short side is the mirror image.

## Pipeline
| file | what it does |
|---|---|
| `src/download.py` | pulls 5m klines, 5m OI (`metrics`) and funding from `data.binance.vision`, then merges them into `data/ETHUSDT_5m.parquet` |
| `src/liqmap.py` | Coinglass-style **model** liquidation map, rebuilt bar by bar with no look-ahead. New OI is opened at the typical price, split long/short by taker-buy share and spread over leverage cohorts (5/10/25/50/100x). Liquidation prices are `p·(1∓1/L±mmr)`. Clusters are removed when touched, shrink on voluntary OI drops, and decay with a 7-day half-life. |
| `src/strategy.py` | the sweep → reclaim state machine, execution with fees and slippage, and stats |
| `src/backtest.py` | baseline; IS grid search; best IS config evaluated **out-of-sample**; **control** on random levels at matched distances; breakdowns by weekend, year and exit reason |
| `src/synth.py` | synthetic data, used only for pipeline tests |

### Rules (long; defaults)
1. Candidate level L = strongest long cluster 1–6% below price with ≥ 5% of the modelled book. Refreshed every hour while idle.
2. **Sweep**: the bar low trades below L. If price extends more than 3% beyond L, the setup is cancelled as a breakdown.
3. **Reclaim**: within 12 bars (1h), a bar closes above L·1.001, **and** OI fell ≥ 0.5% from its pre-sweep high.
4. Entry at the next bar open. Stop 0.1% beyond the sweep extreme. TP at the strongest short cluster above if it is ≥ 1R away, otherwise 2R. Time stop 24h.
5. Costs: 0.05% taker fee plus 0.02% slippage per side. If stop and TP fall in the same bar, the stop is assumed to fill first.

### Why the control matters
Public heatmap levels are seen by everyone. The control runs **identical rules** on levels placed at
random distances, drawn from the real clusters' distance distribution. If the real map doesn't beat
most control seeds, the edge (if any) comes from the reclaim pattern, not from the liquidation map.

## Run
```bash
pip install -r requirements.txt
python -m src.download --start 2022-01          # ~1,800 small zips, a few minutes
python -m src.backtest --data data/ETHUSDT_5m.parquet --out results --split 2025-01-01
```
Outputs in `results/`: `report.json`, `grid_is.csv`, `trades_*.csv`, `equity_baseline.png`.

Robustness: funding filter, 15m/1h reclaim close, zero costs and fixed targets, each with IS, OOS and a random-level control:
```bash
python -m src.robustness --data data/ETHUSDT_5m.parquet --out results --split 2025-01-01
```
A snapshot of the outputs is committed in `docs/`.

Pipeline smoke test (no network):
```bash
python -m src.synth && python -m src.backtest --data data/SYNTH_5m.parquet --out results_synth --split 2024-05-15 --seeds 5 --min-trades 10
```

## Caveats
* This is a **modelled** map. It is not the exchange's real position book, it only covers Binance, and it doesn't see on-chain (Aave) liquidations.
* Binance `metrics` (5m OI) start in Dec 2021, so the sample begins there.
* Choose by IS t-stat, then judge **only** on OOS and on the control comparison.
