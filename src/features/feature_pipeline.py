"""Feature pipeline — orchestrates all indicator transformations.

``FeaturePipeline`` is the single entry-point for converting a clean
OHLCV DataFrame into a feature matrix ready for model training.

Design notes
------------
* **Open/Closed Principle**: add new feature groups by extending the
  ``_STEPS`` list — no existing method needs modification.
* **Single Responsibility**: the pipeline only composes transformations;
  it does not fetch data or train models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.data.preprocessing import add_log_returns
from src.features.technical_indicators import (
    add_bollinger_bands,
    add_ema,
    add_lag_features,
    add_macd,
    add_rolling_volatility,
    add_rsi,
    add_sma,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeaturePipeline:
    """Compose and apply a sequence of feature-engineering transformations.

    Attributes:
        sma_windows:        Windows (days) for Simple Moving Averages.
        ema_spans:          Spans for Exponential Moving Averages.
        rsi_period:         Look-back period for RSI.
        bb_window:          Rolling window for Bollinger Bands.
        lag_offsets:        Lag offsets (days) for lag features.
        volatility_windows: Rolling windows for realised volatility.
    """

    sma_windows: list[int] = field(default_factory=lambda: [7, 14, 30])
    ema_spans: list[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    bb_window: int = 20
    lag_offsets: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    volatility_windows: list[int] = field(default_factory=lambda: [7, 14])

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature-engineering steps to *df*.

        Steps applied in order:
        1. Log returns
        2. SMA
        3. EMA
        4. RSI
        5. MACD
        6. Bollinger Bands
        7. Lag features
        8. Rolling volatility
        9. Drop rows with any remaining NaN values

        Args:
            df: Clean OHLCV DataFrame (output of
                :func:`~src.data.preprocessing.clean_ohlcv`).

        Returns:
            Feature-complete DataFrame with NaN rows removed.
        """
        logger.info("Building features from %d rows of OHLCV data.", len(df))

        steps: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
            add_log_returns,
            lambda d: add_sma(d, windows=self.sma_windows),
            lambda d: add_ema(d, spans=self.ema_spans),
            lambda d: add_rsi(d, period=self.rsi_period),
            add_macd,
            lambda d: add_bollinger_bands(d, window=self.bb_window),
            lambda d: add_lag_features(d, lags=self.lag_offsets),
            lambda d: add_rolling_volatility(d, windows=self.volatility_windows),
        ]

        for step in steps:
            df = step(df)

        n_before = len(df)
        df = df.dropna()
        n_after = len(df)

        logger.info(
            "Feature build complete: %d rows retained (dropped %d NaN rows). "
            "Total features: %d.",
            n_after,
            n_before - n_after,
            len(df.columns),
        )
        return df

    @property
    def feature_columns(self) -> list[str]:
        """Return the list of feature column names produced by :meth:`build`.

        Note: This list is computed deterministically from the pipeline
        configuration.  It does **not** include the target column
        (``close``) or raw OHLCV columns used purely as inputs.
        """
        cols: list[str] = []
        cols.append("log_return")
        cols += [f"sma_{w}" for w in self.sma_windows]
        cols += [f"ema_{s}" for s in self.ema_spans]
        cols.append(f"rsi_{self.rsi_period}")
        cols += ["macd", "macd_signal", "macd_hist"]
        cols += ["bb_upper", "bb_mid", "bb_lower", "bb_width"]
        cols += [f"close_lag_{lag}" for lag in self.lag_offsets]
        cols += [f"volatility_{w}" for w in self.volatility_windows]
        return cols

    @property
    def feature_version(self) -> str:
        """Deterministic fingerprint of this pipeline's configuration + output shape.

        Changes whenever the indicator set, windows, or column order
        change — the minimum bar for "feature versions are recorded"
        (timing-audit Phase 17 / Phase-2 production-safety checklist).
        This is a content fingerprint, not a semantic version number: it
        has no notion of "newer" or "older", only "same" or "different",
        which is exactly what's needed to detect train/live feature skew.
        """
        payload = {
            "sma_windows": self.sma_windows,
            "ema_spans": self.ema_spans,
            "rsi_period": self.rsi_period,
            "bb_window": self.bb_window,
            "lag_offsets": self.lag_offsets,
            "volatility_windows": self.volatility_windows,
            "feature_columns": self.feature_columns,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest[:12]
