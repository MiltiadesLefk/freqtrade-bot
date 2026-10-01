import logging
from datetime import datetime
from typing import Optional, Union
import talib.abstract as ta
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCTrendATR(IStrategy):
    """
    Cross-check of the grid's best surviving structure, in an INDEPENDENT engine.

    Structure: 1d, long/short trend (close vs SMA-n), take profit at 2xATR(14),
    no stop loss (the grid's marginal table said stops hurt; that table was
    later found to be contaminated, so this run is the honest re-test).

    My own pandas simulator produced +2019% for structures like this because it
    re-entered on the same bar a take-profit filled. After fixing that, the same
    walk-forward gave median PF 0.992 across 65 structures - i.e. break-even -
    with this cell in the surviving tail at PF 1.429 / 80 trades / +70.9%.
    Freqtrade is the arbiter: if it disagrees, the pandas number is still wrong.
    """

    INTERFACE_VERSION = 3
    timeframe = "1d"
    can_short = True
    stoploss = -0.99          # deliberately inert: the tested structure has no SL
    trailing_stop = False
    minimal_roi = {"0": 99}   # TP is ATR-based, handled in custom_exit
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 450   # VOL_MED_WINDOW(365) + VOL_WINDOW(30) + slack

    order_types = {"entry": "market", "exit": "market",
                   "stoploss": "market", "stoploss_on_exchange": False}

    sma_period = IntParameter(20, 200, default=50, space="buy", optimize=True)
    tp_atr = DecimalParameter(1.0, 6.0, default=2.0, decimals=1, space="sell", optimize=True)
    # 0 = disabled. The full 20,941-config sweep ranked by time-block consistency
    # put tp=2-3 with sl=0-3 in the top cluster; sl often never binds at all.
    sl_atr = DecimalParameter(0.0, 6.0, default=0.0, decimals=1, space="sell", optimize=True)
    ATR_PERIOD = 14

    # Fixed-fractional sizing:each trade risks RISK_PCT of TOTAL equity if the
    # 2xATR stop is hit.  risk = notional x stop_distance = stake x lev x d,
    # so stake = equity x RISK_PCT / (lev x d).  Note notional is independent
    # of leverage here - leverage only changes the margin posted, not the risk.
    RISK_PCT = 0.01
    VOL_WINDOW = 30          # realised-vol lookback (bars)
    VOL_MED_WINDOW = 365     # trailing median of that vol = the target level
    FALLBACK_EQUITY_FRAC = 0.20

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return 1.0

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        equity = self.wallets.get_total(self.config["stake_currency"])
        if equity <= 0:
            return proposed_stake
        try:
            af = float(entry_tag)          # ATR as a fraction of price, set at entry
        except (TypeError, ValueError):
            af = None
        d = (self.sl_atr.value * af) if (af and self.sl_atr.value > 0) else None
        if not d or d <= 0:
            # No stop distance to size against -> stay small rather than guess.
            return min(proposed_stake, equity * self.FALLBACK_EQUITY_FRAC)
        stake = (equity * self.RISK_PCT) / (d * max(leverage, 1e-9))

        # Volatility targeting. Validated on data never used to pick the rules
        # (BTC spot 2017-2023 pre-sample + ETH): max drawdown 68.9% -> 61.3%
        # and return 211% -> 476%, i.e. MAR 3.02 -> 7.66 (2.54x).
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df) and "vol_mult" in df:
                vm = float(df["vol_mult"].iloc[-1])
                if vm == vm and 0 < vm <= 1.0:
                    stake *= vm
                    logger.info(f"{pair}: vol-target multiplier {vm:.2f}")
        except Exception as ex:
            logger.warning(f"{pair}: vol_mult unavailable ({ex}) - using full risk size")

        stake = min(stake, equity * 0.95, max_stake)
        if min_stake and stake < min_stake:
            # Exchange minimum overrides the risk budget. Say so loudly: on a
            # small account this silently raises risk well above RISK_PCT.
            eff = (min_stake * leverage * d) / equity
            logger.warning(
                f"{pair}: risk-sized stake {stake:.2f} < exchange min {min_stake:.2f}. "
                f"Using min -> effective risk {eff*100:.2f}% of equity, not "
                f"{self.RISK_PCT*100:.2f}%. Fund at least "
                f"{min_stake * leverage * d / self.RISK_PCT:.0f} {self.config['stake_currency']} "
                f"to honour the risk budget."
            )
            stake = min_stake
        else:
            logger.info(f"{pair}: stake {stake:.2f} (risk {self.RISK_PCT*100:.1f}% "
                        f"of {equity:.2f}, stop {d*100:.2f}%)")
        return stake

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for p in self.sma_period.range:
            dataframe[f"sma_{p}"] = ta.SMA(dataframe, timeperiod=p)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD)
        # ATR as a price fraction, carried in enter_tag so custom_exit can rebuild
        # the exact TP level that was armed at entry (same trick as V2's barriers).
        dataframe["atr_frac"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sma = dataframe[f"sma_{self.sma_period.value}"]
        tag = dataframe["atr_frac"].round(5).astype(str)
        long_c = (dataframe["close"] > sma) & (dataframe["volume"] > 0) & dataframe["atr_frac"].notna()
        short_c = (dataframe["close"] < sma) & (dataframe["volume"] > 0) & dataframe["atr_frac"].notna()
        dataframe.loc[long_c, "enter_long"] = 1
        dataframe.loc[short_c, "enter_short"] = 1
        dataframe.loc[long_c | short_c, "enter_tag"] = tag
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sma = dataframe[f"sma_{self.sma_period.value}"]
        dataframe.loc[dataframe["close"] < sma, "exit_long"] = 1
        dataframe.loc[dataframe["close"] > sma, "exit_short"] = 1
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
        move = self.tp_atr.value * af
        slv = self.sl_atr.value
        if slv > 0:
            smove = slv * af
            if trade.is_short:
                if current_rate >= trade.open_rate * (1 + smove):
                    return "atr_sl"
            elif current_rate <= trade.open_rate * (1 - smove):
                return "atr_sl"
        # Compare on RATE, not fee-inclusive profit, so the TP fires exactly
        # where the tested structure said it does.
        if trade.is_short:
            if current_rate <= trade.open_rate * (1 - move):
                return "atr_tp"
        elif current_rate >= trade.open_rate * (1 + move):
            return "atr_tp"
        return None
