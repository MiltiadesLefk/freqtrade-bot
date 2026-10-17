# Results — backtest evidence

Raw evidence behind the strategy research in the [main README](../README.md).

**Every ML (FreqAI) configuration lost money over any window long enough to mean something.**
The only positive ML-family rows are a 2-day, 6-trade probe (`bt_probe`, +1.97%) and a
47-trade, 2-month `TMLXFvg` run (+0.23%, PF 1.02); that is noise, not an edge, and it reversed
on the full span (`TMLXFvg`, 181 trades, -13.16%). The value of this folder is the **method**:
out-of-sample testing, look-ahead bug detection, and cost/turnover analysis. It is not profit.
Nothing here is financial advice; everything ran on paper or in backtests.

## How it was generated

- Source: `/root/freqtrade/ft_userdata/user_data/backtest_results` (Freqtrade backtest output).
- A read-only scan on 2026-10-17 found 120 result files and produced
  [backtest_summary.md](backtest_summary.md) / [backtest_summary.csv](backtest_summary.csv),
  one row per run. The scan only reads; it never alters the result files.
- The `.zip` files in [backtest_results/](backtest_results/) are the raw Freqtrade results
  (result JSON, config, strategy source, market-change feather), each with a `.meta.json`.
- Before publishing, the API-server fields `jwt_secret_key`, `ws_token` and `username` inside the
  configs in those zips were replaced with `REDACTED`. Nothing else was changed.

## Layout

| Path | Contents |
|---|---|
| `backtest_summary.md` / `.csv` | One row per run (120), same data in two formats |
| `backtest_results/` | Raw result zips + `.meta.json` |
| `logs/` | Run logs (downloads, grid sweeps, fullspan runs, A/B runs) |
| `models/` | FreqAI model directories, kept under their `bt_*` identifier |

## Summary columns

`strategy`, `freqaimodel_raw` / `model_learner` (FreqAI learner), `freqai_identifier`
(the `bt_*` id), `timeframe`, `timeframe_detail`, `timerange_code`, `period_start` / `period_end`,
`backtest_days`, `total_trades`, `profit_factor`, `total_profit_pct` / `total_profit_abs`,
`stake_currency`, `max_drawdown_pct`, `win_rate_pct`, `sharpe`, `sortino`, `calmar`,
`trading_mode`, `margin_mode`, `leverage_setting`, `fee_setting` (`n/a` where the saved config
does not store it), `max_open_trades`, `starting_balance`, `final_balance`, `window_type`
(`unknown` where the file does not say), `matched_log_file`, `source_file`, `file_mtime`,
`suspicious_flags`. In the `.md`, model/notes and flags are merged into *Model/learner* and *Notes*.

**Suspicious runs.** The scan flags runs it cannot trust at face value: a trade count far above
the median (~80) or a very low trade count (low statistical significance). These flags are kept in
`suspicious_flags` / the Notes column; 9 of the 120 rows carry one. A flag means "treat with
care", not "proven wrong".

## Models (`models/bt_*`)

What each directory actually holds (from its `run_params.json` and the matching summary rows).
All are BTC/USDT:USDT, 15m, ETH/USDT:USDT as correlated pair, 48-candle label period, 90-day
training window, 2-day backtest period. The `backtesting_predictions/` files are per-retrain
predictions.

| Dir | Learner / strategy in summary | Contents |
|---|---|---|
| `bt_bull_lgbm` | LightGBM, `TradingMLStrategyV2`, 2024-04-01 to 2024-12-31 | 137 prediction files |
| `bt_bull_xgb` | XGBoost, `TradingMLStrategyV2`, 2024-04-01 to 2024-12-31 | 137 prediction files |
| `bt_f1` | XGBoost, `TMLXFlow`, 2025-10-15 to 2026-07-14 | 136 prediction files |
| `bt_f1autopsy` | XGBoost, `TMLXFlow`, 2026-07-12 to 2026-07-14 (4 trades) | 1 prediction file + one trained model: `model.joblib`, feature/label pipelines, and `trained_df.pkl` (cached training features/labels) |
| `bt_f1probe` | XGBoost, `TMLXFlow`, 2026-07-12 to 2026-07-14 (4 trades) | 1 prediction file + `run_params.json` and a small metadata json (no trained model) |
| `bt_fix` | LightGBM, `TradingMLStrategyV2`, 2026-07-16 to 2026-09-06 | 26 prediction files |
| `bt_probe` | LightGBM, `TradingMLStrategyV2`, 2026-07-12 to 2026-07-14 (6 trades) | 1 prediction file + one trained model with `trained_df.pkl` |
| `bt_v3` | LightGBM, 15m `TradingMLStrategyV2` rows (the id says v3, the saved params are 15m) | 136 prediction files |
| `bt_x7` | XGBoost, `TradingMLStrategyV2` and `TMLXFvg` | 136 prediction files |

`bt_f1autopsy` and `bt_probe` (~35 MB each) include cached BTC feature data in `*_trained_df.pkl`
and the fitted pipelines.

**Models are for inspection only.** Reusing one requires the same Freqtrade/FreqAI version
(2026.3 image `freqtradeorg/freqtrade:stable_freqai`), the same strategy feature code and the
same feature set; otherwise loading or predicting will fail or silently misalign. They are not
trading models.
