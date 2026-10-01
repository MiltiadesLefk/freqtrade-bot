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
    stoploss_from_open,
)
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class TradingMLStrategyV3(IStrategy):
    """
    TradingMLStrategyV3 — 1h triple-barrier classifier (RESEARCH, not live)

    V2.1 post-mortem (2026-07-08): three model generations on 15m all failed —
    even a 300-epoch in-sample hyperopt could not find a profitable threshold.
    Diagnosis was economic, not statistical: avg win ~+1.9% (leveraged) vs
    ~0.3% round-trip taker fees at 3-4 trades/day. Fees taxed every win ~16%.

    V3 attacks the fee-to-edge ratio directly: same triple-barrier design on
    1h candles, 48h horizon, ~2x wider barriers. Target win ~+7% leveraged →
    fees drop to ~4% of the target. A few trades per week instead of per day.

    Barrier math vs V2: ATR(24) on 1h ≈ 2x ATR(96) on 15m in price space
    (candle ranges scale ~sqrt(t)), so K=4 here ≈ double V2's barrier width.
    Typical b ≈ 2-3% price → TP/floor ≈ 6-9% leveraged at 3x.

    Run with:  --freqaimodel LightGBMClassifier and a config overlay that sets
    timeframe 1h + include_timeframes [1h, 4h] (see bt_v3_1h.json).
    """

    BARRIER_K = 4.0
    ATR_PERIOD = 24        # 24h of 1h candles — vol basis for the barrier
    REGIME_WINDOW = 168    # 7 days of 1h candles — vol-regime median + donchian
    LEVERAGE = 3.0

    # ── Strategy flags ───────────────────────────────────────────────────────
    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    startup_candle_count = 260  # regime 168+24=192, ema200, + shifts headroom
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
    # Backstop only (leveraged space: -0.24 = -8% price at 3x). Wider than V2
    # because barriers are ~2x wider; floor_ok keeps every floor inside it.
    stoploss = -0.24
    trailing_stop = False
    minimal_roi = {"0": 99}

    # ── FreqUI chart plotting ─────────────────────────────────────────────────
    plot_config = {
        "main_plot": {
            "barrier_upper": {"color": "#26a69a"},
            "barrier_lower": {"color": "#ef5350"},
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
    prob_threshold = DecimalParameter(0.40, 0.75, default=0.50, decimals=2, space="buy", optimize=True)
    prob_margin = DecimalParameter(0.00, 0.45, default=0.15, decimals=2, space="buy", optimize=True)
    exit_prob_threshold = DecimalParameter(0.45, 0.90, default=0.65, decimals=2, space="sell", optimize=True)
    vol_regime_max = DecimalParameter(1.10, 3.00, default=1.50, decimals=2, space="buy", optimize=True)
    # Cooldowns in hours — scaled up for the slower cadence.
    reentry_cooldown_hours = CategoricalParameter(
        [0, 2, 4, 8, 12, 24], default=4, space="sell", optimize=True
    )
    lock_hours = CategoricalParameter(
        [0, 4, 8, 12, 24, 48], default=12, space="sell", optimize=True
    )

    @staticmethod
    def stoploss_space() -> List:
        from freqtrade.optimize.space import SKDecimal
        return [SKDecimal(-0.35, -0.12, decimals=2, name="stoploss")]

    # ── Leverage ──────────────────────────────────────────────────────────────
    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag, side: str, **kwargs) -> float:
        return self.LEVERAGE

    # ── Informative: funding rate candles ─────────────────────────────────────
    def informative_pairs(self):
        return [(pair, "1h", "funding_rate") for pair in self.dp.current_whitelist()]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _label_candles(self) -> int:
        return int(self.freqai_info["feature_parameters"]["label_period_candles"])

    @staticmethod
    def _barrier_from_tag(trade: Trade) -> Optional[float]:
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
            return None
        floor = stoploss_from_open(
            -(b * trade.leverage), current_profit,
            is_short=trade.is_short, leverage=trade.leverage,
        )
        return floor or None

    # ── Custom exit: TP barrier (rate-based) + time barrier ───────────────────
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float,
                    **kwargs) -> Optional[Union[str, bool]]:
        b = self._barrier_from_tag(trade)
        if b is not None:
            if trade.is_short:
                if current_rate <= trade.open_rate * (1 - b):
                    return "tp_barrier"
            elif current_rate >= trade.open_rate * (1 + b):
                return "tp_barrier"

        max_minutes = self._label_candles() * timeframe_to_minutes(self.timeframe)
        if (current_time - trade.open_date_utc) >= timedelta(minutes=max_minutes):
            return "time_barrier"
        return None

    # ── Feature engineering ───────────────────────────────────────────────────
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
        # Long-lookback regime features on the base (1h) timeframe only.
        dataframe["%-roc_2d"] = dataframe["close"].pct_change(48)
        dataframe["%-roc_4d"] = dataframe["close"].pct_change(96)
        dataframe["%-roc_7d"] = dataframe["close"].pct_change(168)

        roll_max = dataframe["close"].rolling(self.REGIME_WINDOW).max()
        roll_min = dataframe["close"].rolling(self.REGIME_WINDOW).min()
        dataframe["%-donchian_pos_7d"] = (
            (dataframe["close"] - roll_min) / (roll_max - roll_min)
        )

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        atr_frac = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD) / dataframe["close"]
        dataframe["%-vol_regime"] = atr_frac / atr_frac.rolling(self.REGIME_WINDOW).median()

        # Funding rate (8h events) — backward merge_asof, no lookahead.
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
        dataframe["&-tb_label"] = labels
        return dataframe

    # ── Indicators ────────────────────────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["barrier_frac"] = (
            self.BARRIER_K
            * ta.ATR(dataframe, timeperiod=self.ATR_PERIOD)
            / dataframe["close"]
        )
        dataframe["vol_regime"] = dataframe["barrier_frac"] / (
            dataframe["barrier_frac"].rolling(self.REGIME_WINDOW).median()
        )
        dataframe["barrier_upper"] = dataframe["close"] * (1 + dataframe["barrier_frac"])
        dataframe["barrier_lower"] = dataframe["close"] * (1 - dataframe["barrier_frac"])
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

    # ── Post-exit cooldowns (order_filled fires for ALL exits incl. exchange
    # stops, in both live and backtest — see V2.1 audit) ──────────────────────
    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime,
                     **kwargs) -> None:
        if trade.is_open or order.ft_order_side == trade.entry_side:
            return
        is_stop = order.ft_order_side == "stoploss" or trade.exit_reason in (
            "stoploss", "stoploss_on_exchange", "trailing_stop_loss", "liquidation",
        )
        hours = self.lock_hours.value if is_stop else self.reentry_cooldown_hours.value
        if hours and hours > 0:
            reason = "stoploss_cooldown" if is_stop else "reentry_cooldown"
            self.lock_pair(
                pair,
                until=current_time + timedelta(hours=float(hours)),
                reason=f"{reason}_{hours}h",
            )
