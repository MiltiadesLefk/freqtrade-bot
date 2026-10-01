import logging

import pandas as pd
from pandas import DataFrame

from TradingMLStrategyV2 import TradingMLStrategyV2

logger = logging.getLogger(__name__)


class TMLXBasis(TradingMLStrategyV2):
    """
    X1 research variant (2026-07-14): V2 + perpetual basis features.

    Basis = (last price − mark price) / mark price on 1h candles — the
    instantaneous premium aggressive longs are paying (funding is the same
    signal integrated over 8h). New information the model has never seen;
    everything else inherits from V2 unchanged.
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return (
            [(p, "1h", "funding_rate") for p in pairs]
            + [(p, "1h", "mark") for p in pairs]
        )

    def feature_engineering_standard(self, dataframe: DataFrame,
                                     metadata: dict, **kwargs) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)

        cols = ["%-basis", "%-basis_z_7d"]
        mark = last = None
        try:
            mark = self.dp.get_pair_dataframe(metadata["pair"], "1h", candle_type="mark")
            last = self.dp.get_pair_dataframe(metadata["pair"], "1h")
        except Exception as e:
            logger.warning(f"Basis fetch failed for {metadata['pair']}: {e}")
        if (mark is not None and last is not None
                and not mark.empty and not last.empty):
            m = mark[["date", "close"]].rename(columns={"close": "mark_close"})
            px = last[["date", "close"]].rename(columns={"close": "last_close"})
            b = pd.merge(px, m, on="date", how="inner").sort_values("date")
            b["%-basis"] = (b["last_close"] - b["mark_close"]) / b["mark_close"]
            mu = b["%-basis"].rolling(168, min_periods=24).mean()
            sd = b["%-basis"].rolling(168, min_periods=24).std()
            b["%-basis_z_7d"] = (b["%-basis"] - mu) / sd
            dataframe = pd.merge_asof(
                dataframe, b[["date"] + cols], on="date", direction="backward"
            )
        else:
            logger.warning(f"No mark-price data for {metadata['pair']} — basis stays NaN")
            for c in cols:
                dataframe[c] = float("nan")
        return dataframe
