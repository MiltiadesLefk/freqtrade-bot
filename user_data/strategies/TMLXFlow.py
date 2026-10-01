import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from TradingMLStrategyV2 import TradingMLStrategyV2

logger = logging.getLogger(__name__)

FLOW_DIR = Path("/freqtrade/user_data/flowdata")


class TMLXFlow(TradingMLStrategyV2):
    """
    F1 (2026-07-16): NEW-INFORMATION campaign — order-flow + positioning.

    Adds features the model has never seen, from Binance raw dumps (not in
    freqtrade's store): per-candle taker buy/sell imbalance and CVD (aggressor
    flow), open-interest dynamics, and long/short positioning ratios.
    Everything else identical to V2. Compare full-span vs the XGBoost
    full-span baseline (−27.64%) — same learner, same span, features only.

    Lookahead notes: taker volume belongs to its own candle row (same
    convention as volume — known at candle close). Metrics rows are
    point-in-time snapshots at create_time; merge_asof backward, no shift.
    """

    _flow = None
    _metrics = None

    def _load_flow(self):
        if TMLXFlow._flow is None:
            TMLXFlow._flow = pd.read_feather(FLOW_DIR / "flow_15m.feather")
            TMLXFlow._metrics = pd.read_feather(FLOW_DIR / "metrics_5m.feather")
        return TMLXFlow._flow, TMLXFlow._metrics

    def feature_engineering_standard(self, dataframe: DataFrame,
                                     metadata: dict, **kwargs) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)
        try:
            flow, metrics = self._load_flow()
        except Exception as e:
            logger.warning(f"flow data unavailable: {e}")
            return dataframe

        # ── taker flow (joined on the candle's own date) ──
        f = flow.copy()
        delta = 2 * f["taker_buy_volume"] - f["volume"]
        f["%-flow_imb"] = delta / f["volume"].replace(0, pd.NA)
        f["%-cvd_24h"] = delta.rolling(96).sum() / f["volume"].rolling(96).sum()
        f["%-cvd_7d"] = delta.rolling(672).sum() / f["volume"].rolling(672).sum()
        dataframe = dataframe.merge(
            f[["date", "%-flow_imb", "%-cvd_24h", "%-cvd_7d"]], on="date", how="left"
        )

        # ── OI / positioning (backward asof; snapshots known at create_time) ──
        mcols = ["%-oi_roc_1h", "%-oi_roc_24h", "%-oi_z_7d", "%-taker_ls", "%-top_ls_z"]
        m = metrics.rename(columns={
            "oi_roc_1h": "%-oi_roc_1h", "oi_roc_24h": "%-oi_roc_24h",
            "oi_z_7d": "%-oi_z_7d", "taker_ls": "%-taker_ls", "top_ls_z": "%-top_ls_z",
        })[["date"] + mcols].sort_values("date")
        dataframe = pd.merge_asof(
            dataframe.sort_values("date"), m, on="date", direction="backward"
        )

        # ── OI-price confirmation: new-longs vs short-covering regimes ──
        price_roc_1h = dataframe["close"].pct_change(4)
        dataframe["%-oi_price_conf"] = dataframe["%-oi_roc_1h"] * price_roc_1h * 1e4
        return dataframe
