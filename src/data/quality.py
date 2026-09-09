"""Data-quality gate for OHLCV candle data (Phase 14 of the Phase-2 validation brief).

"A bad data point should not silently become a model prediction." Before
this module, nothing between an exchange-adapter fetch and
``ModelManager.predict_from_frame`` ever checked whether the candles that
came back were even sane — a duplicated timestamp, a `high < low` row (bad
data or a provider bug), a multi-candle gap from a dropped connection, or a
suspiciously stale last candle would all have flowed straight into feature
engineering and out the other end as a confident-looking prediction.

This module is intentionally dependency-free (pure pandas/numpy) so it can
sit directly in the hot path of live inference without adding fragility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.timeframes import timeframe_to_seconds


@dataclass
class DataQualityIssue:
    severity: str  # "critical" | "warning"
    code: str
    message: str


@dataclass
class DataQualityReport:
    issues: list[DataQualityIssue] = field(default_factory=list)

    @property
    def is_safe_to_use(self) -> bool:
        """False if any *critical* issue was found. Warnings alone don't block use."""
        return not any(i.severity == "critical" for i in self.issues)

    @property
    def critical_issues(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    def to_dict(self) -> dict:
        return {
            "is_safe_to_use": self.is_safe_to_use,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message}
                for i in self.issues
            ],
        }


class DataQualityError(ValueError):
    """Raised when a critical data-quality issue makes a frame unsafe to use."""

    def __init__(self, report: DataQualityReport):
        self.report = report
        messages = "; ".join(i.message for i in report.critical_issues)
        super().__init__(f"Data quality gate failed: {messages}")


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    now: datetime | None = None,
    max_stale_candles: float = 2.0,
    max_missing_fraction: float = 0.05,
    volume_zscore_threshold: float = 8.0,
    is_closed: pd.Series | None = None,
) -> DataQualityReport:
    """Run all data-quality checks against an OHLCV frame.

    Args:
        df:            OHLCV frame, ascending ``DatetimeIndex`` (UTC), columns
                       ``[open, high, low, close, volume]``.
        timeframe:     Candle period string, used to compute the expected
                       candle spacing.
        now:           Current time (UTC) for the staleness check. Defaults
                       to ``datetime.now(timezone.utc)``.
        max_stale_candles: How many candle-intervals old the last candle may
                       be before it's flagged as stale (critical) — e.g.
                       ``2.0`` on a 1H timeframe allows up to ~2h of lag.
        max_missing_fraction: Fraction of expected candles that may be
                       missing across the whole series before it's a
                       critical issue (a handful of gaps is a warning;
                       a large fraction means the series can't be trusted).
        volume_zscore_threshold: Volume z-score beyond which a candle is
                       flagged as an abnormal-volume warning.
        is_closed:     Optional boolean Series aligned to ``df.index``
                       (e.g. carried over from ``CandleBar.is_closed``
                       before it was dropped by converting to a plain
                       OHLCV frame). ``src/utils/candles.py::bars_to_frame``
                       already drops a trailing unclosed candle at the
                       point live/backtest data is first assembled — this
                       is a *second*, independent gate for any caller that
                       builds an OHLCV frame by another path (e.g. a
                       concatenation of two fetched batches, or a frame
                       reloaded from a provenance-tracked CSV that kept
                       its own closure metadata) and therefore isn't
                       covered by that single enforcement point. Any
                       ``False`` entry — not just a trailing one — is
                       critical: an unclosed candle anywhere in a frame
                       used for research or a prediction anchor is unsafe.

    Returns:
        A :class:`DataQualityReport`. Does not raise — use
        :func:`enforce_quality_gate` when a hard failure is desired.
    """
    issues: list[DataQualityIssue] = []

    if df.empty:
        issues.append(DataQualityIssue("critical", "empty_frame", "OHLCV frame is empty."))
        return DataQualityReport(issues)

    required_cols = {"open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        issues.append(
            DataQualityIssue(
                "critical", "missing_columns", f"Missing required columns: {sorted(missing_cols)}"
            )
        )
        return DataQualityReport(issues)

    # --- duplicate candles -------------------------------------------------
    n_duplicates = int(df.index.duplicated().sum())
    if n_duplicates:
        issues.append(
            DataQualityIssue(
                "critical", "duplicate_candles", f"{n_duplicates} duplicate timestamp(s) found."
            )
        )

    # --- ordering ------------------------------------------------------------
    if not df.index.is_monotonic_increasing:
        issues.append(
            DataQualityIssue("critical", "unsorted_index", "OHLCV index is not sorted ascending.")
        )

    # --- impossible OHLC values ---------------------------------------------
    numeric = df[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        issues.append(
            DataQualityIssue("critical", "non_finite_values", "NaN or infinite OHLCV value(s) found.")
        )
    if (numeric[["open", "high", "low", "close"]] <= 0).to_numpy().any():
        issues.append(
            DataQualityIssue("critical", "non_positive_price", "Zero or negative price value(s) found.")
        )
    impossible = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    n_impossible = int(impossible.sum())
    if n_impossible:
        issues.append(
            DataQualityIssue(
                "critical",
                "impossible_ohlc",
                f"{n_impossible} candle(s) violate high >= max(open,close,low) and/or "
                f"low <= min(open,close,high).",
            )
        )
    if (df["volume"] < 0).any():
        issues.append(
            DataQualityIssue("critical", "negative_volume", "Negative volume value(s) found.")
        )

    # --- missing candles / timestamp gaps -----------------------------------
    interval_seconds = timeframe_to_seconds(timeframe)
    if len(df) >= 2:
        # Note: pandas' DatetimeIndex storage resolution is not guaranteed
        # to be nanoseconds (pandas >= 2.x can use us/ms/s depending on how
        # the index was built), so dividing a raw int64 view by a
        # hard-coded 1e9 silently gives the wrong answer on some pandas
        # versions. Diffing as timedelta64 and dividing by a timedelta64
        # unit is resolution-independent.
        deltas = np.diff(df.index.to_numpy()) / np.timedelta64(1, "s")
        expected_span = interval_seconds * (len(df) - 1)
        actual_span = float(df.index[-1].timestamp() - df.index[0].timestamp())
        implied_total_candles = actual_span / interval_seconds + 1 if interval_seconds else len(df)
        missing_fraction = (
            max(0.0, (implied_total_candles - len(df)) / implied_total_candles)
            if implied_total_candles > 0
            else 0.0
        )
        n_gaps = int((deltas > interval_seconds * 1.5).sum())
        if n_gaps:
            severity = "critical" if missing_fraction > max_missing_fraction else "warning"
            issues.append(
                DataQualityIssue(
                    severity,
                    "missing_candles",
                    f"{n_gaps} timestamp gap(s) larger than one candle interval "
                    f"(~{missing_fraction:.1%} of expected candles missing overall).",
                )
            )

    # --- staleness -----------------------------------------------------------
    now = now or datetime.now(timezone.utc)
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    age_seconds = (now - last_ts.to_pydatetime()).total_seconds()
    if interval_seconds and age_seconds > interval_seconds * max_stale_candles:
        issues.append(
            DataQualityIssue(
                "critical",
                "stale_data",
                f"Last candle is {age_seconds / 3600:.1f}h old — more than "
                f"{max_stale_candles} candle-intervals stale for timeframe '{timeframe}'.",
            )
        )

    # --- unclosed candles ------------------------------------------------------
    if is_closed is not None:
        aligned = is_closed.reindex(df.index)
        if aligned.isna().any():
            issues.append(
                DataQualityIssue(
                    "critical",
                    "is_closed_alignment_mismatch",
                    "is_closed series does not cover every row in the OHLCV frame's index.",
                )
            )
        else:
            n_unclosed = int((~aligned.astype(bool)).sum())
            if n_unclosed:
                issues.append(
                    DataQualityIssue(
                        "critical",
                        "unclosed_candle",
                        f"{n_unclosed} candle(s) are not yet closed (is_closed=False) — "
                        "an in-progress candle must never be used for research or as a prediction anchor.",
                    )
                )

    # --- abnormal volume (warning only -- unusual volume can be legitimate) --
    if len(df) >= 20:
        vol = df["volume"].astype(float)
        mean, std = float(vol.mean()), float(vol.std())
        if std > 0:
            z = (vol - mean) / std
            n_outliers = int((z.abs() > volume_zscore_threshold).sum())
            if n_outliers:
                issues.append(
                    DataQualityIssue(
                        "warning",
                        "abnormal_volume",
                        f"{n_outliers} candle(s) with volume >{volume_zscore_threshold} "
                        f"standard deviations from the mean.",
                    )
                )

    return DataQualityReport(issues)


def enforce_quality_gate(df: pd.DataFrame, timeframe: str, **kwargs: object) -> DataQualityReport:
    """Run :func:`validate_ohlcv` and raise :class:`DataQualityError` on any
    critical issue. Warnings are returned, not raised, so callers can log
    them without blocking a prediction over e.g. a single unusual-volume
    candle.
    """
    report = validate_ohlcv(df, timeframe, **kwargs)
    if not report.is_safe_to_use:
        raise DataQualityError(report)
    return report
