import logging
from typing import List, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    CategoricalParameter,
    stoploss_from_absolute,
)
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class TradingMLStrategyV2(IStrategy):
    """
    TradingMLStrategyV2 — BTC-Only FreqAI Triple-Barrier Classifier

    Redesign after the 2026-07-06 audit showed the 3h-return regressor had no
    directional edge (win rate flat ~28% across prediction strength).

    Approach (López de Prado triple-barrier):
      - Label each 15m candle by which barrier price touches FIRST within the
        next 48 candles (12h): +4×ATR(96) → "tb_up", −4×ATR(96) → "tb_down",
        neither → "tb_flat".  Class balance on 2025-07→2026-07 data:
        ~31% / 34% / 34% — near-balanced, vol-adaptive, drift-resistant.
      - LightGBMClassifier outputs P(tb_up), P(tb_down), P(tb_flat) columns.
      - Enter when the directional probability clears prob_threshold AND beats
        the opposite class by prob_margin, with do_predict == 1.
      - Execution mirrors the label exactly: TP at +barrier, floor at −barrier,
        time-exit at 12h. What the model predicts is what the trade does.

    Run with:  --freqaimodel LightGBMClassifier
    Leverage 3x (down from 5x). All custom_* profit values are in leveraged
    profit space; barrier fractions stored in enter_tag are in PRICE space.
    """

    # Barrier width in ATR(96) multiples — must match the label definition.
    BARRIER_K = 4.0
    ATR_PERIOD = 96  # 24h of 15m candles
    LEVERAGE = 3.0

    # ── Strategy flags ───────────────────────────────────────────────────────
    timeframe = "15m"
    can_short = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    startup_candle_count = 800  # vol_regime needs 672 (median) + 96 (ATR) = 768
    use_custom_stoploss = True

    # ── Orders ───────────────────────────────────────────────────────────────
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    # ── Risk defaults ─────────────────────────────────────────────────────────
    # Backstop only (leveraged profit space: -0.15 ≈ -5% price at 3x).
    # The real stop is the per-trade barrier floor in custom_stoploss.
    stoploss = -0.15
    trailing_stop = False
    minimal_roi = {"0": 99}

    # ── FreqUI chart plotting ─────────────────────────────────────────────────
    # barrier_upper / barrier_lower are the ±barrier envelope around price. At an
    # entry candle they equal that trade's actual TP and SL levels: for a LONG,
    # upper = TP / lower = SL; for a SHORT, mirror it (lower = TP / upper = SL).
    # FreqUI can't auto-draw the TP because it lives in custom_exit, not on the
    # exchange — this makes it visible like the stoploss line.
    plot_config = {
        "main_plot": {
            # Flat, TradingView-position-style lines per trade (live/dry only):
            # tp_line / sl_line span each trade from entry to exit at the
            # actual armed levels. barrier_upper/lower is the rolling ±b
            # envelope — what TP/SL WOULD be if a trade opened on that candle.
            "tp_line": {"color": "#26a69a", "type": "line"},
            "sl_line": {"color": "#ef5350", "type": "line"},
            "barrier_upper": {"color": "#1b5e57", "type": "line"},
            "barrier_lower": {"color": "#7f3a37", "type": "line"},
        },
        "subplots": {
            "Model probability": {
                "tb_up": {"color": "#26a69a"},
                "tb_down": {"color": "#ef5350"},
                "tb_flat": {"color": "#888888"},
            },
            "Vol regime (entries paused above threshold)": {
                "vol_regime": {"color": "#f0b90b"},
            },
        },
    }

    # ── Tunables ──────────────────────────────────────────────────────────────
    # Minimum probability of the directional class to enter
    prob_threshold = DecimalParameter(0.40, 0.75, default=0.45, decimals=2, space="buy", optimize=True)
    # Directional probability must beat the opposite class by this much
    prob_margin = DecimalParameter(0.00, 0.45, default=0.10, decimals=2, space="buy", optimize=True)
    # Opposite-class probability needed to force an early exit — deliberately
    # higher than the entry bar; otherwise the model churns out of trades
    # before either barrier is reached and fees eat the account.
    exit_prob_threshold = DecimalParameter(0.45, 0.90, default=0.60, decimals=2, space="sell", optimize=True)
    # Volatility circuit breaker: no NEW entries while ATR(96) exceeds this
    # multiple of its own 7-day median. The model's probabilities are
    # calibrated on the trailing 90d regime; when vol doubles overnight
    # (Feb 2026: 69% ann. vol vs 40% trained) it gets steamrolled.
    vol_regime_max = DecimalParameter(1.10, 3.00, default=1.50, decimals=2, space="buy", optimize=True)
    # Hours to pause after ANY exit (not just losses) — throttles fee churn.
    reentry_cooldown_hours = CategoricalParameter(
        [0, 0.5, 1.0, 2.0, 3.0, 4.0], default=2.0, space="sell", optimize=True
    )
    # Hours to lock the pair after a stoploss/floor exit
    lock_hours = CategoricalParameter(
        [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        default=3.0, space="sell", optimize=True,
    )

    @staticmethod
    def stoploss_space() -> List:
        from freqtrade.optimize.space import SKDecimal
        return [SKDecimal(-0.25, -0.08, decimals=2, name="stoploss")]

    # ── Leverage ──────────────────────────────────────────────────────────────
    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag, side: str, **kwargs) -> float:
        return self.LEVERAGE

    # ── Position sizing: fixed-fractional risk of the FULL account ────────────
    # stake = (equity × RISK_PCT) / (barrier_frac × leverage): hitting the
    # barrier floor loses ≈ RISK_PCT of total equity on every trade, however
    # wide the vol-dependent barrier happens to be. Wider barrier → smaller
    # stake, tighter barrier → larger stake, same dollar risk either way.
    # tradable_balance_ratio stays ~0.95 (fee/funding buffer only) — this
    # method owns real position size, and reported profit percentages are
    # honest against the whole account instead of a pre-carved 25% pool.
    RISK_PCT = 0.010
    MAX_STAKE_EQUITY_FRAC = 0.50   # cap: a near-zero barrier can't blow up the stake
    FALLBACK_STAKE_EQUITY_FRAC = 0.25  # no parseable barrier → only the -0.15
    # backstop bounds the loss, so keep the stake small (≤ 3.75% of equity)

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        equity = self.wallets.get_total(self.config["stake_currency"])
        if equity <= 0:
            return proposed_stake
        try:
            b = float(entry_tag) if entry_tag else None
        except (TypeError, ValueError):
            b = None
        if b is None or not (0 < b < 0.2):
            return min(proposed_stake, equity * self.FALLBACK_STAKE_EQUITY_FRAC)
        stake = (equity * self.RISK_PCT) / (b * leverage)
        stake = min(stake, equity * self.MAX_STAKE_EQUITY_FRAC, max_stake)
        if min_stake:
            stake = max(stake, min_stake)
        return stake

    # ── Informative: funding rate candles ─────────────────────────────────────
    def informative_pairs(self):
        return [(pair, "1h", "funding_rate") for pair in self.dp.current_whitelist()]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _label_candles(self) -> int:
        return int(self.freqai_info["feature_parameters"]["label_period_candles"])

    @staticmethod
    def _barrier_from_tag(trade: Trade) -> Optional[float]:
        """Barrier fraction (price space) stored in enter_tag at signal time."""
        try:
            b = float(trade.enter_tag)
            return b if 0 < b < 0.2 else None
        except (ValueError, TypeError):
            return None

    # ── Custom stoploss: barrier floor pinned to open ─────────────────────────
    def custom_stoploss(self, pair: str, trade: Trade,
                        current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        b = self._barrier_from_tag(trade)
        if b is None:
            return None  # keep backstop
        # Pin the floor to the label's exact barrier PRICE. stoploss_from_open
        # is not equivalent: freqtrade passes current_profit fee- and
        # funding-inclusive, which displaced every floor ~0.10% price tighter
        # than the label barrier (all 19 live trades, audit 2026-07-14).
        floor_price = trade.open_rate * ((1 + b) if trade.is_short else (1 - b))
        floor = stoploss_from_absolute(
            floor_price, current_rate,
            is_short=trade.is_short, leverage=trade.leverage,
        )
        # 0.0 (= rate already beyond floor) is discarded by freqtrade anyway;
        # the ratcheted stop from the after-fill call already covers that case.
        return floor or None

    # ── Custom exit: TP barrier + time barrier ────────────────────────────────
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float,
                    **kwargs) -> Optional[Union[str, bool]]:
        b = self._barrier_from_tag(trade)
        if b is not None:
            # Trigger on RATE touching the +barrier level, exactly like the
            # label. current_profit is fee-inclusive: comparing it against
            # b×leverage demands an extra ~2×fee move beyond the barrier, so
            # trades that touch the barrier exactly never took profit.
            if trade.is_short:
                if current_rate <= trade.open_rate * (1 - b):
                    return "tp_barrier"
            elif current_rate >= trade.open_rate * (1 + b):
                return "tp_barrier"

        max_minutes = self._label_candles() * timeframe_to_minutes(self.timeframe)
        if (current_time - trade.open_date_utc) >= timedelta(minutes=max_minutes):
            return "time_barrier"
        return None

    # ── Feature engineering (lean + stationary) ───────────────────────────────
    def feature_engineering_expand_all(self, dataframe: DataFrame,
                                       period: int, metadata: dict,
                                       **kwargs) -> DataFrame:
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)

        bollinger = ta.BBANDS(dataframe, timeperiod=period)
        dataframe[f"%-bb_width-period_{period}"] = (
            bollinger["upperband"] - bollinger["lowerband"]
        ) / bollinger["middleband"]
        dataframe[f"%-bb_pctb-period_{period}"] = (
            (dataframe["close"] - bollinger["lowerband"]) /
            (bollinger["upperband"] - bollinger["lowerband"])
        )

        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-atr_pct-period_{period}"] = (
            ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        )
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame,
                                         metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-adx"] = ta.ADX(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe)
        dataframe["%-macdhist_norm"] = macd["macdhist"] / dataframe["close"]

        dataframe["%-volume_ratio"] = (
            dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        )
        dataframe["%-high_low_range"] = (
            dataframe["high"] - dataframe["low"]
        ) / dataframe["close"]

        ema10 = ta.EMA(dataframe, timeperiod=10)
        ema50 = ta.EMA(dataframe, timeperiod=50)
        ema200 = ta.EMA(dataframe, timeperiod=200)
        dataframe["%-ema_10_50_ratio"] = (ema10 - ema50) / ema50
        dataframe["%-ema_50_200_ratio"] = (ema50 - ema200) / ema200
        dataframe["%-ema_50_dist"] = (dataframe["close"] - ema50) / dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame,
                                     metadata: dict, **kwargs) -> DataFrame:
        # Long-lookback regime features on the base timeframe ONLY — putting
        # these in expand_basic would recompute them on 1h too, where a 672-
        # candle lookback needs 4 weeks of data and NaN-starves predictions.
        dataframe["%-roc_48h"] = dataframe["close"].pct_change(192)
        dataframe["%-roc_96h"] = dataframe["close"].pct_change(384)
        dataframe["%-roc_7d"] = dataframe["close"].pct_change(672)

        roll_max = dataframe["close"].rolling(self.ATR_PERIOD).max()
        roll_min = dataframe["close"].rolling(self.ATR_PERIOD).min()
        dataframe["%-donchian_pos_24h"] = (
            (dataframe["close"] - roll_min) / (roll_max - roll_min)
        )

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        # Vol regime — same ratio the entry circuit-breaker uses. The model
        # should see the state that gates its own trades.
        atr_frac = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD) / dataframe["close"]
        dataframe["%-vol_regime"] = atr_frac / atr_frac.rolling(672).median()

        # Funding rate (8h events on Binance) — positioning/crowding signal
        # invisible to OHLCV. merge_asof backward: each 15m candle only sees
        # the most recent PAST funding event, so no lookahead.
        fr_cols = ["%-funding_rate", "%-funding_3d_avg", "%-funding_z_30d"]
        fr = None
        try:
            fr = self.dp.get_pair_dataframe(
                metadata["pair"], "1h", candle_type="funding_rate"
            )
        except Exception as e:
            logger.warning(f"Funding-rate fetch failed for {metadata['pair']}: {e}")
        if fr is not None and not fr.empty:
            fr = fr[["date", "open"]].rename(columns={"open": "fr"}).sort_values("date")
            fr["%-funding_rate"] = fr["fr"]
            fr["%-funding_3d_avg"] = fr["fr"].rolling(9, min_periods=3).mean()
            fr_mean = fr["fr"].rolling(90, min_periods=30).mean()
            fr_std = fr["fr"].rolling(90, min_periods=30).std()
            fr["%-funding_z_30d"] = (fr["fr"] - fr_mean) / fr_std
            dataframe = pd.merge_asof(
                dataframe, fr[["date"] + fr_cols], on="date", direction="backward"
            )
        else:
            logger.warning(
                f"No funding-rate data for {metadata['pair']} — features stay NaN"
            )
            for col in fr_cols:
                dataframe[col] = np.nan
        return dataframe

    # ── Triple-barrier target ─────────────────────────────────────────────────
    def set_freqai_targets(self, dataframe: DataFrame,
                           metadata: dict, **kwargs) -> DataFrame:
        H = self._label_candles()
        close = dataframe["close"].values
        high = dataframe["high"].values
        low = dataframe["low"].values
        atr_frac = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD).values / close
        barrier = self.BARRIER_K * atr_frac

        n = len(dataframe)
        labels = np.full(n, np.nan, dtype=object)
        for i in range(n - H):
            b = barrier[i]
            if not np.isfinite(b):
                continue
            up_lvl = close[i] * (1 + b)
            dn_lvl = close[i] * (1 - b)
            w_hi = high[i + 1:i + 1 + H]
            w_lo = low[i + 1:i + 1 + H]
            hit_u = w_hi >= up_lvl
            hit_d = w_lo <= dn_lvl
            u = np.argmax(hit_u) if hit_u.any() else 10 ** 6
            d = np.argmax(hit_d) if hit_d.any() else 10 ** 6
            if u < d:
                labels[i] = "tb_up"
            elif d < u:
                labels[i] = "tb_down"
            else:
                labels[i] = "tb_flat"
        # Tail rows (future unknown) stay NaN → dropped from training.
        dataframe["&-tb_label"] = labels
        return dataframe

    # ── Indicators ────────────────────────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Barrier width at signal time — consumed by enter_tag / exits.
        dataframe["barrier_frac"] = (
            self.BARRIER_K
            * ta.ATR(dataframe, timeperiod=self.ATR_PERIOD)
            / dataframe["close"]
        )
        # Vol-regime ratio: current 24h ATR vs its 7-day median.
        dataframe["vol_regime"] = dataframe["barrier_frac"] / (
            dataframe["barrier_frac"].rolling(672).median()
        )
        # Visualization-only barrier envelope for FreqUI (see plot_config).
        dataframe["barrier_upper"] = dataframe["close"] * (1 + dataframe["barrier_frac"])
        dataframe["barrier_lower"] = dataframe["close"] * (1 - dataframe["barrier_frac"])

        # Flat per-trade TP/SL lines for FreqUI (TradingView position style).
        # Live/dry only: backtest plotting has its own trade overlay, and
        # LocalTrade vs Trade semantics differ there.
        dataframe["tp_line"] = np.nan
        dataframe["sl_line"] = np.nan
        if self.dp and self.dp.runmode.value in ("live", "dry_run"):
            try:
                tf_min = timeframe_to_minutes(self.timeframe)
                window_start = dataframe["date"].iloc[0]
                for t in Trade.get_trades_proxy(pair=metadata["pair"]):
                    b = self._barrier_from_tag(t)
                    if b is None or t.open_date_utc is None:
                        continue
                    end = t.close_date_utc or dataframe["date"].iloc[-1]
                    if end < window_start:
                        continue
                    start = pd.Timestamp(t.open_date_utc).floor(f"{tf_min}min")
                    mask = (dataframe["date"] >= start) & (dataframe["date"] <= end)
                    tp = t.open_rate * ((1 - b) if t.is_short else (1 + b))
                    sl = t.stop_loss or t.open_rate * ((1 + b) if t.is_short else (1 - b))
                    dataframe.loc[mask, "tp_line"] = tp
                    dataframe.loc[mask, "sl_line"] = sl
            except Exception as e:
                logger.warning(f"tp/sl plot lines skipped: {e}")
        return dataframe

    # ── Entries ───────────────────────────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame,
                             metadata: dict) -> DataFrame:
        p = self.prob_threshold.value
        m = self.prob_margin.value

        p_up = dataframe["tb_up"].iloc[-1]
        p_dn = dataframe["tb_down"].iloc[-1]
        do_predict = dataframe["do_predict"].iloc[-1]
        logger.info(
            f"🔎 {metadata['pair']} | P(up)={p_up:.2f} P(down)={p_dn:.2f} "
            f"do_predict={do_predict} | need p>{p} margin>{m}"
        )

        calm = dataframe["vol_regime"] < self.vol_regime_max.value
        # The barrier floor must sit INSIDE the hard backstop, otherwise the
        # backstop stops the trade out before the label's down-barrier and
        # execution no longer matches what the model was trained to predict.
        floor_ok = (
            dataframe["barrier_frac"].notna()
            & ((dataframe["barrier_frac"] * self.LEVERAGE) < abs(self.stoploss))
        )
        long_cond = (
            (dataframe["do_predict"] == 1)
            & calm
            & floor_ok
            & (dataframe["tb_up"] > p)
            & ((dataframe["tb_up"] - dataframe["tb_down"]) > m)
            & (dataframe["volume"] > 0)
        )
        short_cond = (
            (dataframe["do_predict"] == 1)
            & calm
            & floor_ok
            & (dataframe["tb_down"] > p)
            & ((dataframe["tb_down"] - dataframe["tb_up"]) > m)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[short_cond, "enter_short"] = 1
        dataframe.loc[long_cond | short_cond, "enter_tag"] = (
            dataframe["barrier_frac"].round(5).astype(str)
        )
        return dataframe

    # ── Exits (model flips) ───────────────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        p = self.exit_prob_threshold.value
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["tb_down"] > p),
            "exit_long",
        ] = 1
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["tb_up"] > p),
            "exit_short",
        ] = 1
        return dataframe

    # ── Post-exit cooldowns ───────────────────────────────────────────────────
    # Locking must live in order_filled, NOT confirm_trade_exit: exchange
    # stoploss fills happen on Binance and never pass through
    # confirm_trade_exit, so a lock placed there silently skips exactly the
    # exits that need a cooldown most (verified live 2026-07-07: stop-out at
    # 15:36 produced no lock). Do NOT gate on trade.is_open here: in
    # backtesting order_filled fires BEFORE trade.close(), so an is_open
    # guard silently disables every lock in every backtest (audit 2026-07-14:
    # median exit→re-entry gap in the holdout was one candle).
    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime,
                     **kwargs) -> None:
        if order.ft_order_side == trade.entry_side:
            return
        is_stop = order.ft_order_side == "stoploss" or trade.exit_reason in (
            "stop_loss", "stoploss_on_exchange", "trailing_stop_loss", "liquidation",
        )
        hours = self.lock_hours.value if is_stop else self.reentry_cooldown_hours.value
        if hours and hours > 0:
            reason = "stoploss_cooldown" if is_stop else "reentry_cooldown"
            self.lock_pair(
                pair,
                until=current_time + timedelta(hours=float(hours)),
                reason=f"{reason}_{hours}h",
            )
