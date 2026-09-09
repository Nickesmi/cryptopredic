"""Historical, point-in-time investable-universe reconstruction (Phase 6, Section 3).

"At every historical timestamp T, the investable universe must contain
only assets that were actually available at T. Do NOT use today's
Binance symbol list for historical periods." Phase 5's synthetic
universes enforced this via an all-NaN prefix on a shared padded
DatetimeIndex (cheap and correct for generated data of known length).
Real per-asset histories will not share one index — different assets
list on different dates, get fetched with different amounts of history,
and may have their own internal gaps. This module reconstructs
membership directly from each asset's own index instead of relying on a
shared-frame NaN trick, so it works on genuinely ragged real data.

Two-tier listing-date semantics, kept explicit and separate everywhere
in this module:

- **Data-derived** ("proxy"): an asset's own first available OHLCV
  timestamp from whatever source fetched it. This is NOT the same as
  the asset's real exchange listing date -- a data source may simply not
  have earlier history cached, even though the asset traded earlier.
- **Registry-derived** ("verified"): an explicit listing (and, where
  known, delisting) date from an actual exchange symbol-status log or
  equivalent registry. This sandbox has no such registry (Section 1) --
  every listing date used anywhere in Phase 6 is therefore a **proxy**,
  and every function below refuses to silently upgrade a proxy into a
  verified fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AssetListingRecord:
    symbol: str
    first_available_timestamp: pd.Timestamp
    source: str
    is_proxy: bool  # True: inferred from data's own first candle, not a verified listing date
    last_available_timestamp: pd.Timestamp | None = None  # None if still tradable / unknown
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "first_available_timestamp": str(self.first_available_timestamp),
            "source": self.source,
            "is_proxy": self.is_proxy,
            "last_available_timestamp": (
                str(self.last_available_timestamp) if self.last_available_timestamp is not None else None
            ),
            "notes": self.notes,
        }


def infer_listing_registry_from_data(
    asset_histories: dict[str, pd.DataFrame], source: str = "ohlcv_first_candle_proxy"
) -> dict[str, AssetListingRecord]:
    """Build a *proxy* listing registry from each asset's own first non-NaN OHLCV candle.

    Every record is marked ``is_proxy=True`` -- this is explicitly NOT a
    substitute for a real exchange listing-date registry (Section 3), only
    the best mechanically-derivable lower bound this sandbox can compute
    on its own.
    """
    registry: dict[str, AssetListingRecord] = {}
    for symbol, df in asset_histories.items():
        valid = df.dropna(how="all")
        if valid.empty:
            continue
        registry[symbol] = AssetListingRecord(
            symbol=symbol,
            first_available_timestamp=valid.index.min(),
            source=source,
            is_proxy=True,
            last_available_timestamp=valid.index.max(),
            notes="Lower bound only: the asset may have traded before this data source's earliest cached candle.",
        )
    return registry


def eligible_symbols_at(
    timestamp: pd.Timestamp,
    asset_histories: dict[str, pd.DataFrame],
    listing_registry: dict[str, AssetListingRecord] | None = None,
) -> list[str]:
    """Which symbols were actually tradable/data-available at *timestamp*.

    A symbol is eligible iff:
    1. its own OHLCV history has a non-NaN row at or before *timestamp*, AND
    2. if a *listing_registry* entry exists for it, *timestamp* is not
       before that record's ``first_available_timestamp`` (a registry can
       only make eligibility *stricter* than the raw data suggests, never
       looser -- it cannot manufacture history the data doesn't have).
    """
    eligible: list[str] = []
    for symbol, df in asset_histories.items():
        valid = df.dropna(how="all")
        if valid.empty or valid.index.min() > timestamp:
            continue
        if listing_registry is not None and symbol in listing_registry:
            record = listing_registry[symbol]
            if timestamp < record.first_available_timestamp:
                continue
            if record.last_available_timestamp is not None and timestamp > record.last_available_timestamp:
                continue
        eligible.append(symbol)
    return sorted(eligible)


def build_point_in_time_universe(
    asset_histories: dict[str, pd.DataFrame],
    scan_timestamps: list[pd.Timestamp],
    listing_registry: dict[str, AssetListingRecord] | None = None,
) -> dict[pd.Timestamp, list[str]]:
    """``{timestamp: [eligible symbols]}`` for every timestamp in *scan_timestamps*.

    This is the point-in-time universe Phase 6 Section 3's diagram
    describes: ``timestamp -> assets actually tradable at timestamp ->
    historical OHLCV available by timestamp``. The third step (slicing
    each eligible asset's OHLCV to ``<= timestamp``) is the caller's
    job -- this function only answers *which* symbols belong at each
    point, matching how ``src/research/walk_forward.py`` and Phase 4's
    ``cross_sectional_analysis.py`` already separate "who's in the
    universe" from "what does their history look like so far".
    """
    return {ts: eligible_symbols_at(ts, asset_histories, listing_registry) for ts in scan_timestamps}


@dataclass
class SurvivorshipReport:
    quantifiable: bool
    n_current_universe: int
    n_known_delisted: int | None = None
    delisted_fraction: float | None = None
    message: str = ""
    delisted_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantifiable": self.quantifiable,
            "n_current_universe": self.n_current_universe,
            "n_known_delisted": self.n_known_delisted,
            "delisted_fraction": self.delisted_fraction,
            "message": self.message,
            "delisted_symbols": self.delisted_symbols,
        }


def quantify_survivorship_gap(
    current_universe_symbols: set[str],
    delisted_registry: list[str] | None = None,
) -> SurvivorshipReport:
    """Quantify survivorship bias -- honestly refusing to invent a number without real data.

    Phase 6 Section 3: "If delisted assets cannot be reconstructed,
    explicitly quantify the survivorship limitation. Do not silently hide
    it." The only way to *quantify* a bias from delisted assets is to
    actually have a list of them (a real exchange delisting log); without
    one, the correct behaviour is to say so plainly (``quantifiable=False``)
    rather than substitute a guess, a "typical" industry figure, or a
    synthetic estimate that would look like real evidence.
    """
    if delisted_registry is None:
        return SurvivorshipReport(
            quantifiable=False,
            n_current_universe=len(current_universe_symbols),
            message=(
                "No historical delisting registry is available in this environment (Section 1). "
                "Survivorship bias from delisted/failed assets cannot be quantified -- reporting "
                "this limitation explicitly rather than substituting an estimate."
            ),
        )

    n_delisted = len(delisted_registry)
    n_total = len(current_universe_symbols) + n_delisted
    fraction = n_delisted / n_total if n_total > 0 else None
    return SurvivorshipReport(
        quantifiable=True,
        n_current_universe=len(current_universe_symbols),
        n_known_delisted=n_delisted,
        delisted_fraction=fraction,
        message=f"{n_delisted} known delisted asset(s) out of {n_total} total historically-listed asset(s).",
        delisted_symbols=list(delisted_registry),
    )
