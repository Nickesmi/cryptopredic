"""Generic, signal-agnostic walk-forward evaluation harness (Phase 5, Sections 2, 4, 5, 8).

Deliberately reuses the generic (non-scanner) primitives from
``src/evaluation/cross_sectional_analysis.py`` — ``CrossSectionalSnapshot``,
``forward_return``, ``information_coefficient``, ``compute_ic_series``,
``quintile_analysis``, ``rank_turnover`` — instead of duplicating them.
Those functions only ever consume a plain ``{symbol: score}`` dict; none
of them call ``scan_candidate`` or otherwise touch
``src/ranking/score_coins.py``, so importing them here does not couple
the Alpha Research Engine to the production scanner. What *is* new here
is how the scores get built: ``build_signal_panel`` below scores a
universe from one of ``src/research/signals.py``'s pure signal functions,
not from the scanner.

No-lookahead / no-future-listing guarantee: at scan point ``t``, an asset
contributes a score only if its precomputed signal series has a
non-``NaN`` value at ``t - 1`` (the same "last visible candle" anchor
convention ``forward_return`` uses). Since every signal in
``signals.py`` is a causal rolling/shift transform, and a not-yet-listed
asset's price series is ``NaN`` before its listing candle by
construction in ``src/research/universes.py``, this one NaN check is
simultaneously the warmup-period guard AND the "never introduce
future-listed assets into earlier historical rankings" guard (Section 4)
— an asset that lists later simply has no score, and is excluded, at
every earlier snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.cross_sectional_analysis import (  # noqa: F401 (re-exported for convenience)
    CrossSectionalSnapshot,
    compute_ic_series,
    forward_return,
    information_coefficient,
    quintile_analysis,
    rank_turnover,
)


def build_signal_panel(
    candidates: dict[str, pd.DataFrame],
    signal_series: dict[str, pd.Series],
    scan_every_candles: int,
    min_train_size: int,
    max_scans: int | None = None,
) -> list[CrossSectionalSnapshot]:
    """Score every candidate, at every scan point, from a precomputed causal signal series.

    Args:
        candidates:        ``{symbol: OHLCV DataFrame}``, all sharing the
                            same DatetimeIndex (see ``src/research/universes.py``
                            — assets not yet "listed" carry ``NaN`` rows
                            rather than a shorter index).
        signal_series:     ``{symbol: pd.Series}`` aligned to each
                            candidate's index — the output of one
                            ``SignalSpec.series(...)`` call per symbol.
        scan_every_candles: Stride between scan points.
        min_train_size:    Candles of history required before the first
                            scan point (independent of any one signal's
                            own warmup — a signal still not warmed up by
                            then is simply excluded per-snapshot via the
                            NaN check below).
        max_scans:         Optional cap on the number of scan points.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    index_len = len(next(iter(candidates.values())))
    for symbol, df in candidates.items():
        if len(df) != index_len:
            raise ValueError(
                f"All candidates must share the same index length (got {len(df)} for "
                f"'{symbol}', expected {index_len}). Not-yet-listed assets should be "
                "represented as NaN rows on a shared index, not a shorter index."
            )

    scan_points = list(range(min_train_size, index_len, scan_every_candles))
    if max_scans is not None:
        scan_points = scan_points[:max_scans]

    panel: list[CrossSectionalSnapshot] = []
    for t in scan_points:
        anchor_idx = t - 1
        scores: dict[str, float] = {}
        excluded: list[str] = []
        for symbol, series in signal_series.items():
            value = series.iloc[anchor_idx]
            if pd.isna(value):
                excluded.append(symbol)
            else:
                scores[symbol] = float(value)

        first_symbol = next(iter(candidates))
        scan_time = int(candidates[first_symbol].index[anchor_idx].timestamp())
        panel.append(CrossSectionalSnapshot(scan_index=t, scan_time=scan_time, scores=scores, excluded=excluded))

    return panel


def restrict_panel_to_range(panel: list[CrossSectionalSnapshot], start_idx: int, end_idx: int) -> list[CrossSectionalSnapshot]:
    """Keep only snapshots whose scan_index falls in [start_idx, end_idx)."""
    return [snap for snap in panel if start_idx <= snap.scan_index < end_idx]


@dataclass(frozen=True)
class RegionSplit:
    design: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


def split_design_validation_test(
    n: int, design_frac: float = 0.5, validation_frac: float = 0.25
) -> RegionSplit:
    """Deterministic, index-based three-way split of a length-*n* time index.

    Section 5: "Design/Development", "Validation", "Frozen Test" regions.
    Purely positional (no shuffling — this is a time series), and the
    boundaries are fixed once by this function; nothing in this package
    should ever compute them differently for different experiments, or
    the "frozen test" guarantee (touched at most once) has no meaning.
    """
    if not (0 < design_frac < 1) or not (0 < validation_frac < 1) or design_frac + validation_frac >= 1:
        raise ValueError("design_frac and validation_frac must be in (0,1) and sum to < 1")
    design_end = int(n * design_frac)
    validation_end = int(n * (design_frac + validation_frac))
    return RegionSplit(design=(0, design_end), validation=(design_end, validation_end), test=(validation_end, n))


def cross_sectional_relative_strength_panel(
    candidates: dict[str, pd.DataFrame],
    scan_every_candles: int,
    min_train_size: int,
    lookback: int,
    max_scans: int | None = None,
) -> list[CrossSectionalSnapshot]:
    """Section 2G/2E: rank each asset by (its trailing return - the equal-weight universe's trailing return).

    Built directly (not via ``build_signal_panel``) because this signal is
    intrinsically cross-sectional: it needs every asset's trailing return
    at the same snapshot to compute the universe average, which a
    single-asset precomputed series cannot supply.
    """
    trailing_return: dict[str, pd.Series] = {
        symbol: df["close"].pct_change(lookback) for symbol, df in candidates.items()
    }
    index_len = len(next(iter(candidates.values())))
    scan_points = list(range(min_train_size, index_len, scan_every_candles))
    if max_scans is not None:
        scan_points = scan_points[:max_scans]

    panel: list[CrossSectionalSnapshot] = []
    for t in scan_points:
        anchor_idx = t - 1
        raw: dict[str, float] = {}
        for symbol, series in trailing_return.items():
            value = series.iloc[anchor_idx]
            if not pd.isna(value):
                raw[symbol] = float(value)

        excluded = [s for s in candidates if s not in raw]
        if raw:
            universe_mean = float(np.mean(list(raw.values())))
            scores = {s: v - universe_mean for s, v in raw.items()}
        else:
            scores = {}

        first_symbol = next(iter(candidates))
        scan_time = int(candidates[first_symbol].index[anchor_idx].timestamp())
        panel.append(CrossSectionalSnapshot(scan_index=t, scan_time=scan_time, scores=scores, excluded=excluded))

    return panel
