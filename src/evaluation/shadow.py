"""Shadow-mode evaluation loop (Phase 13 of the Phase-2 validation brief).

"Shadow mode" means: keep generating and storing predictions without
executing any trades, then automatically score each one once its horizon
expires. ``PredictionStore`` already had every piece needed for this
(``create_prediction``, ``due_predictions``, ``evaluate_prediction``) —
what was missing, called out explicitly in the first audit pass, is
anything that actually *calls* ``evaluate_prediction`` once a prediction's
``expires_at`` has passed. Nothing in the codebase closed that loop.

This module closes it: :func:`evaluate_due_predictions` pulls every
expired-but-unevaluated prediction and scores it against a real price
looked up at (or immediately after) its target timestamp. The price
lookup is injected (``price_lookup``) rather than hard-coded to Binance so
this function is fully unit-testable without network access — which
matters here specifically because this sandbox cannot reach Binance or
CoinGecko (see docs/TIMING_AUDIT_REPORT.md and the Phase-2 report's
Section on network access). :func:`binance_price_lookup` is the real
production implementation; it has not been exercised against live data in
this environment for that reason, only against a stub in tests.

Deployment note: this module provides the evaluation *function*, not a
scheduler. Running it on a timer (cron, Celery beat, a simple `while True`
+ sleep loop) is an infrastructure decision left to deployment — see
``scripts/run_shadow_cycle.py`` for a minimal periodic entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from src.evaluation.prediction_store import PredictionStore


class PriceLookup(Protocol):
    async def __call__(self, symbol: str, timeframe: str, at_time: int) -> tuple[float, int]:
        """Return ``(actual_price, actual_time)`` for *symbol* at or immediately
        after unix timestamp *at_time*."""
        ...


@dataclass
class ShadowCycleResult:
    n_due: int
    n_evaluated: int
    n_errors: int
    evaluated_prediction_ids: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_due": self.n_due,
            "n_evaluated": self.n_evaluated,
            "n_errors": self.n_errors,
            "evaluated_prediction_ids": self.evaluated_prediction_ids,
            "errors": self.errors,
        }


async def evaluate_due_predictions(
    store: PredictionStore,
    price_lookup: Callable[[str, str, int], Awaitable[tuple[float, int]]],
    now_ts: int | None = None,
) -> ShadowCycleResult:
    """Evaluate every prediction whose horizon has expired but hasn't been scored yet.

    Args:
        store:        The prediction ledger.
        price_lookup: Async callable returning ``(actual_price, actual_time)``
                      for a symbol at/after a given timestamp. Inject
                      :func:`binance_price_lookup` in production, a stub in
                      tests.
        now_ts:       Override "now" for testability. Defaults to the
                      current time.

    Returns:
        A :class:`ShadowCycleResult` summarising what happened. A failed
        price lookup for one prediction is recorded in ``errors`` and does
        not stop the rest of the batch from being evaluated — one bad
        symbol must not silently freeze the whole feedback loop.
    """
    due = store.due_predictions(now_ts=now_ts)
    evaluated_ids: list[str] = []
    errors: list[dict[str, Any]] = []

    for prediction in due:
        try:
            actual_price, actual_time = await price_lookup(
                prediction["symbol"], prediction["timeframe"], prediction["expires_at"]
            )
            store.evaluate_prediction(
                prediction["prediction_id"],
                actual_time=actual_time,
                actual_price=actual_price,
            )
            evaluated_ids.append(prediction["prediction_id"])
        except Exception as exc:  # noqa: BLE001 -- one bad lookup must not abort the batch
            errors.append({"prediction_id": prediction["prediction_id"], "error": str(exc)})

    return ShadowCycleResult(
        n_due=len(due),
        n_evaluated=len(evaluated_ids),
        n_errors=len(errors),
        evaluated_prediction_ids=evaluated_ids,
        errors=errors,
    )


async def binance_price_lookup(
    symbol: str, timeframe: str, at_time: int, session=None
) -> tuple[float, int]:
    """Production :class:`PriceLookup`: the closed Binance candle covering *at_time*.

    NOTE: not exercised against live data in this repository's CI/sandbox
    environment (no outbound network access here — see the Phase-2
    report). Review carefully and test against a live Binance endpoint
    before relying on it in production.
    """
    from src.data.exchange.factory import ExchangeFactory
    from src.utils.candles import bars_to_frame
    from src.utils.timeframes import timeframe_to_seconds

    adapter = ExchangeFactory.create("binance", session=session)
    bars = await adapter.get_candles(symbol, timeframe, limit=5)
    df = bars_to_frame(bars, drop_unclosed=True)

    interval = timeframe_to_seconds(timeframe)
    # Find the closed candle whose window covers at_time; fall back to the
    # most recent closed candle if at_time is in the future relative to
    # what's available yet (shouldn't happen for a genuinely due
    # prediction, but fail toward "use the latest known price" rather than
    # raising).
    covering = df[df.index.map(lambda ts: int(ts.timestamp()) <= at_time)]
    row = covering.iloc[-1] if not covering.empty else df.iloc[-1]
    actual_time = int(row.name.timestamp()) + interval  # candle close time
    return float(row["close"]), actual_time
