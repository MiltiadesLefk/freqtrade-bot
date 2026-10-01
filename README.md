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

## Layout

```
docker-compose.yml        # runs BTCTrendATR in dry-run by default
.env.example              # copy to .env; secrets are read from env vars
user_data/
  config.json             # V2 / FreqAI config (15m)
  config_trend.json       # BTCTrendATR config (1d)
  strategies/             # all strategies + their .json parameter files
  bt_*.json               # backtest overlays (identifier + one change each)
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
