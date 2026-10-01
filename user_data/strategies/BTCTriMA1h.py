import logging
from datetime import datetime
from typing import Optional, Union
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCTriMA1h(IStrategy):
    """
    Higher-frequency alternative to BTCTrendATR: ~61 trades/yr vs ~33.

    Rule ("cross3"): long only while fast > mid > slow SMA, all aligned.
    slow = n, mid = n/3, fast = n/8, with n=300 on 1h (slow ~12.5d).
    Take profit 3xATR(14), no stop (the alignment breaking is the exit), 1x.

    Selected the same way as BTCTrendATR - consistency across 8 time blocks,
    then ranked by WORST regime - but with a >=60 trades/yr floor. Of 640
    high-frequency configs only 18 were profitable in all three regimes and 5
    cleared worst-regime 1.15, so selection risk here is real and this needs
    the freqtrade + --timeframe-detail confirmation before it means anything.

    pandas sweep: 167 trades, PF 1.398, bull 1.534 / bear 1.435 / side 1.375,
    +100.5%, DD 29.6%.
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    stoploss = -0.99
    trailing_stop = False
    minimal_roi = {"0": 99}
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 400

    order_types = {"entry": "market", "exit": "market",
                   "stoploss": "market", "stoploss_on_exchange": False}

    slow_period = IntParameter(100, 500, default=300, space="buy", optimize=True)
    tp_atr = DecimalParameter(1.0, 8.0, default=3.0, decimals=1, space="sell", optimize=True)
    ATR_PERIOD = 14
    allow_short = True
    RISK_PCT = 0.01
    FALLBACK_EQUITY_FRAC = 0.20

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for n in self.slow_period.range:
            dataframe[f"s_{n}"] = ta.SMA(dataframe, timeperiod=n)
            dataframe[f"m_{n}"] = ta.SMA(dataframe, timeperiod=max(3, n // 3))
            dataframe[f"f_{n}"] = ta.SMA(dataframe, timeperiod=max(2, n // 8))
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD)
        dataframe["atr_frac"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        n = self.slow_period.value
        f, m, s = dataframe[f"f_{n}"], dataframe[f"m_{n}"], dataframe[f"s_{n}"]
        cond = (f > m) & (m > s) & (dataframe["volume"] > 0) & dataframe["atr_frac"].notna()
        scond = (f < m) & (m < s) & (dataframe["volume"] > 0) & dataframe["atr_frac"].notna()
        dataframe.loc[cond, "enter_long"] = 1
        if self.allow_short:
            dataframe.loc[scond, "enter_short"] = 1
        dataframe.loc[cond | scond, "enter_tag"] = dataframe["atr_frac"].round(5).astype(str)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        n = self.slow_period.value
        f, m, s = dataframe[f"f_{n}"], dataframe[f"m_{n}"], dataframe[f"s_{n}"]
        dataframe.loc[~((f > m) & (m > s)), "exit_long"] = 1
        dataframe.loc[~((f < m) & (m < s)), "exit_short"] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float,
                    **kwargs) -> Optional[Union[str, bool]]:
        try:
            af = float(trade.enter_tag)
        except (TypeError, ValueError):
            return None
        if not (0 < af < 0.5):
            return None
        mv = self.tp_atr.value * af
        if trade.is_short:
            if current_rate <= trade.open_rate * (1 - mv):
                return "atr_tp"
        elif current_rate >= trade.open_rate * (1 + mv):
            return "atr_tp"
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        equity = self.wallets.get_total(self.config["stake_currency"])
        if equity <= 0:
            return proposed_stake
        try:
            af = float(entry_tag)
        except (TypeError, ValueError):
            af = None
        # No stop here, so size against the take-profit distance as the risk unit.
        d = (self.tp_atr.value * af) if af else None
        if not d or d <= 0:
            return min(proposed_stake, equity * self.FALLBACK_EQUITY_FRAC)
        stake = (equity * self.RISK_PCT) / (d * max(leverage, 1e-9))
        stake = min(stake, equity * 0.95, max_stake)
        if min_stake and stake < min_stake:
            stake = min_stake
        return stake
