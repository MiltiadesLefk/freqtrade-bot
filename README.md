# freqtrade-bot — BTC perp strategy research

A research log, in code, of trying to build a profitable BTC/USDT perpetual-futures
bot on [Freqtrade](https://www.freqtrade.io) / [FreqAI](https://www.freqtrade.io/en/stable/freqai/).
Everything ran on **paper (dry-run) only**.

The short version: ~40 ML configurations on 15m bars lost money, in bear, chop and bull
regimes alike. A systematic sweep then showed why, and led to a much simpler daily
trend-following bot with a small, *unproven* edge.

> **Not financial advice.** Nothing here is a recommendation to trade. No strategy in
> this repo has been shown to be reliably profitable.

## Strategies

Each strategy was added in the order it was tried (see `git log`).

| Strategy | Idea | Outcome |
|---|---|---|
| `TradingMLStrategyV2` | FreqAI LightGBM classifier, triple-barrier label (±4×ATR96, 12h horizon), 15m, 3x isolated, 1% equity risk per trade | Loses on every long window; a good live week was a calm-regime effect |
| `TradingMLStrategyV3` | Same design on 1h with a 48h horizon and 2× barriers, to cut fee drag | Fees fixed (~24% of V2's loss) but the model was still wrong |
| `TMLXBasis` (X1) | V2 + perp-basis features | Worse: new information hurt |
| X3 overlay | V2 + SOL correlated-pair features (~114 extra features) | Much worse |
| X4 overlay | V2 with a strongly regularised LightGBM | Least bad of the feature/regularisation tweaks, still negative |
| `TMLXFlow` (F1) | V2 + taker buy/sell flow and open-interest features (XGBoost) | Noise-level change vs baseline on a full-span backtest |
| `BTCTrendATR` | **No ML.** 1d, long/short, `close` vs SMA(50), TP/SL at 2×ATR(14), 1x, fixed-fractional sizing | Small edge, not statistically proven (see below) |
| `BTCTriMA1h` | 1h triple-MA cross | Runner-up; PF ≈ 1.0 out of sample once run through the honest engine |

Also tried and rejected without code here: a lean-feature set, a 4h feature suite,
24h/wider-barrier labels, RandomForest/XGBoost/MLP learners, direction filters, ADX/volatility
regime filters, partial take-profits, fee-aware EV gates, liquidity/SMC-style entries and
stops (as ML filters and as standalone strategies), and a 200-epoch hyperopt.

## What I learned

1. **The edge was never absent at 15m — turnover ate it.** A ~159k-config sweep across
   timeframes (1m → 1d), rules, lookbacks, TP/SL and trailing, scored by consistency across
   8 time blocks, found 100 configs that won ≥7/8 blocks on 1d, 108 on 4h, 57 on 1h and
   **0 on 15m / 5m / 1m**. Profit factor rises monotonically with bar size.
2. **More trades are worse out of sample.** Demanding ≥60 trades/year, 4h and 1h
   candidates fell to PF ≈ 1.0 OOS while the 1d bot kept PF ≈ 1.3.
3. **Rank by worst regime, not overall PF.** Bear markets are the binding constraint.
4. **Use `--timeframe-detail` for any TP/SL strategy.** Without it exits fill at the candle
   close, far past the target, and the ranking of strategies reverses.
5. **If most configs win, the simulator is broken.** A resume-on-the-exit-bar bug produced
   +2000% returns and 68/73 winners.
6. **Look-ahead bugs do all the work in "promising" results.** A feature taken from the entry
   bar instead of the last closed bar turned a PF 0.81 meta-labelling filter into PF 1.49.
7. **Single-window improvements are noise.** Two "winning" screens failed full-span validation.
8. **Meta-labelling amplifies an edge; it cannot create one.**
9. **Don't hyperopt daily strategies.** The tuned pick overfit (IS PF 2.46 → OOS 1.22) while
   the consistency-ranked pick generalised better (IS 1.15 → OOS 1.41).

## Honest status of `BTCTrendATR`

- Freqtrade backtest with `--timeframe-detail`: full-window PF ≈ 1.22, OOS PF ≈ 1.33 (92 trades).
- Frozen on unseen data (BTC spot 2017–2023, ETH spot): PF ≈ 1.23 — **about half as good**
  as the selection window suggested. Plan on ~1.2, not 1.5.
- Drawdown on long history was **~67–71%** at full-wallet sizing, vs ~25% in the selection window.
- The Deflated Sharpe Ratio fails at every assumed number of trials: with only ~92 trades the
  Sharpe can't be separated from the best of many correlated trials. That is "not disproven,
  insufficient evidence", not "proven".
- Diversifying across assets diluted the result rather than helping.

## Results

All ML configurations lost money over meaningful windows; the value of this repo is the method
(out-of-sample testing, look-ahead bug detection, cost/turnover analysis), not profit. Full notes:
[results/README.md](results/README.md). Raw zips, logs and FreqAI models are in `results/`.

Below, 42 of the 120 runs, copied unchanged from
[results/backtest_summary.csv](results/backtest_summary.csv): each strategy's worst run, every
run flagged as suspicious (⚠), and the best run of the strategies that had a positive one. The
rest are in the summary file. TF = timeframe, PF = profit factor, DD = drawdown.

| Strategy | TF | Period | Trades | PF | Profit % | Max DD % | Flag |
|---|---|---|---|---|---|---|---|
| TradingMLStrategyOld | 15m | 2026-04-20 to 2026-07-06 | 108 | 0.77 | -9.5 | 9.5 |  |
| TradingMLStrategy | 15m | 2026-04-20 to 2026-07-06 | 237 | 0.62 | -30.45 | 32.88 |  |
| TradingMLStrategyV2 | 15m | 2025-10-15 to 2026-05-15 | 1057 | 0.79 | -45.02 | 46.61 | ⚠ trade count far above median (~80): 1057 |
| TradingMLStrategyV3 | 1h | 2026-05-15 to 2026-07-06 | 30 | 0.62 | -5.04 | 6.07 |  |
| TMLXLong | 15m | 2026-05-15 to 2026-07-14 | 98 | 0.55 | -14.7 | 15.09 |  |
| TMLXShort | 15m | 2026-05-15 to 2026-07-14 | 80 | 0.98 | -0.45 | 6.85 |  |
| TMLXEV6 | 15m | 2026-05-15 to 2026-07-14 | 141 | 0.81 | -8.01 | 11.25 |  |
| TMLXEV9 | 15m | 2026-05-15 to 2026-07-14 | 139 | 0.76 | -10.18 | 13.44 |  |
| TMLXTP60 | 15m | 2026-05-15 to 2026-07-14 | 150 | 0.81 | -8.08 | 10.31 |  |
| TMLXRange25 | 15m | 2026-05-15 to 2026-07-14 | 92 | 0.66 | -10.75 | 14.83 |  |
| TradingMLStrategyV2 | 15m | 2026-07-12 to 2026-07-14 | 6 | 2.73 | 1.97 | 1.13 | ⚠ very low trade count: 6 (low statistical significance) |
| TradingMLStrategyV2 | 15m | 2026-07-12 to 2026-07-14 | 6 | 2.73 | 1.97 | 1.13 | ⚠ very low trade count: 6 (low statistical significance) |
| TMLXBasis | 15m | 2026-05-15 to 2026-07-14 | 139 | 0.67 | -15.03 | 19.12 |  |
| TMLX4h | 15m | 2026-05-15 to 2026-07-14 | 142 | 0.75 | -10.82 | 14.31 |  |
| TMLXK6 | 15m | 2026-05-15 to 2026-07-14 | 47 | 0.34 | -10.35 | 10.64 |  |
| TMLXLiq | 15m | 2026-05-15 to 2026-07-14 | 149 | 0.59 | -17.54 | 21.3 |  |
| TMLXLiq192 | 15m | 2026-05-15 to 2026-07-14 | 139 | 0.72 | -10.33 | 14.32 |  |
| TMLXFvg | 15m | 2026-05-15 to 2026-07-14 | 47 | 1.02 | 0.23 | 5.08 |  |
| TMLXIfvg | 15m | 2026-05-15 to 2026-07-14 | 100 | 0.76 | -7.94 | 10.02 |  |
| TMLXBosFade | 15m | 2026-05-15 to 2026-07-14 | 119 | 0.54 | -17.79 | 19.72 |  |
| TMLXSweep | 15m | 2026-05-15 to 2026-07-14 | 46 | 0.59 | -8.66 | 10.71 |  |
| TMLXPdhTarget | 15m | 2026-05-15 to 2026-07-14 | 134 | 0.63 | -15.79 | 18.09 |  |
| TMLXFvg | 15m | 2025-10-15 to 2026-07-14 | 181 | 0.78 | -13.16 | 17.51 |  |
| TMLXBosAlign | 15m | 2025-10-15 to 2026-07-14 | 394 | 0.79 | -24.71 | 31.36 |  |
| PureFvg | 15m | 2025-10-15 to 2026-07-14 | 26 | 3.17 | 37.08 | 17.07 |  |
| PureBos | 15m | 2025-10-15 to 2026-07-14 | 17 | 4.25 | 42.26 | 13.02 |  |
| PureSweep | 15m | 2025-10-15 to 2026-07-14 | 396 | 0.78 | -36.57 | 44.65 |  |
| PureFvg | 15m | 2025-10-15 to 2026-07-14 | 1564 | 0.84 | -72.79 | 72.99 | ⚠ trade count far above median (~80): 1564 |
| PureBos | 15m | 2025-10-15 to 2026-07-14 | 499 | 0.82 | -43.41 | 44.77 |  |
| TMLXFlow | 15m | 2026-07-12 to 2026-07-14 | 4 | 0.51 | -0.6 | 1.13 | ⚠ very low trade count: 4 (low statistical significance) |
| TMLXFlow | 15m | 2025-10-15 to 2026-07-14 | 431 | 0.83 | -24.9 | 29.87 |  |
| TMLXFlow | 15m | 2026-07-12 to 2026-07-14 | 4 | 0.51 | -0.6 | 1.13 | ⚠ very low trade count: 4 (low statistical significance) |
| TradingMLStrategyV2 | 15m | 2024-04-01 to 2024-12-31 | 401 | 0.63 | -50.69 | 51.19 |  |
| BTCTrendDaily | 1d | 2024-08-07 to 2026-09-05 | 15 | 3.45 | 140.66 | 16.67 |  |
| BTCTrendDaily | 1d | 2024-08-07 to 2026-09-05 | 27 | 0.88 | -13.15 | 47.64 |  |
| BTCTrendDaily | 1d | 2024-08-07 to 2025-06-08 | 4 | 5.21 | 30.63 | 7.28 | ⚠ very low trade count: 4 (low statistical significance) |
| BTCTrendDaily | 1d | 2025-06-09 to 2026-09-05 | 4 | 0.92 | -0.97 | 12.88 | ⚠ very low trade count: 4 (low statistical significance) |
| BTCTrendDaily | 1d | 2024-08-07 to 2025-06-08 | 10 | 3.07 | 45.06 | 9.53 | ⚠ very low trade count: 10 (low statistical significance) |
| BTCTrendATR | 1d | 2025-06-09 to 2026-09-05 | 37 | 0.85 | -14.64 | 35.78 |  |
| BTCTriMA1h | 1h | 2023-12-17 to 2026-09-06 | 213 | 1.25 | 29.13 | 10.81 |  |
| BTCTriMA1h | 1h | 2025-06-09 to 2026-09-06 | 240 | 1.0 | -0.66 | 14.46 |  |
| BTCTrendATR | 1d | 2025-02-23 to 2026-09-05 | 76 | 1.48 | 90.87 | 23.36 |  |

## Layout

```
docker-compose.yml        # runs BTCTrendATR in dry-run by default
.env.example              # copy to .env; secrets are read from env vars
user_data/
  config.json             # V2 / FreqAI config (15m)
  config_trend.json       # BTCTrendATR config (1d)
  strategies/             # all strategies + their .json parameter files
  bt_*.json               # backtest overlays (identifier + one change each)
results/                  # backtest summary, raw result zips, logs, FreqAI models
```

`TMLXFlow` expects pre-processed flow/OI data in `user_data/flowdata/` (built from Binance
Vision dumps); the data-prep script is not included.

## Running

```bash
cp .env.example .env            # fill in values; keep dry_run: true
docker compose up -d            # paper-trade BTCTrendATR

# backtest (download data first with `download-data`)
docker compose run --rm freqtrade backtesting \
  --config user_data/config_trend.json --strategy BTCTrendATR \
  --timeframe 1d --timeframe-detail 1h --timerange 20231201-
```

FreqAI strategies use the experiment overlays, e.g.
`--config user_data/config.json --config user_data/bt_x4.json --strategy TradingMLStrategyV2`.
FreqAI prediction caches are keyed by identifier, so any feature/label/timerange change needs a
fresh `identifier`.

## Requirements

Docker and the `freqtradeorg/freqtrade:stable_freqai` image (Freqtrade 2026.3).
