"""Precise, independent definitions of prediction "success".

Phase 9 of the Phase-2 validation brief is explicit: *"A 4H prediction
should NOT be considered a successful 4H prediction merely because the
target was reached 3 days later."* Before this module, the codebase had
exactly one binary notion of success — ``EvaluationMetrics.direction_correct``
in ``src/evaluation/metrics.py`` — which only ever asks "was the direction
right, evaluated at the promised horizon". It cannot express "the price was
eventually right, just late", which is precisely the failure mode this
whole audit exists to catch.

This module defines five **independent** success criteria. None of them
substitute for another, and a prediction can be true on some and false on
others simultaneously — that is the point:

* **Directional success** — sign(actual return at the promised horizon)
  matches sign(predicted return). Says nothing about magnitude or timing.
* **Price success** — the actual price *at the promised horizon* was
  within ``price_tolerance_pct`` of the predicted price. Says nothing
  about direction being "impressively" right vs. barely, and nothing
  about timing beyond the one fixed evaluation point.
* **Target success** — the predicted price level was reached *at some
  point* within ``target_search_window_seconds`` (default: no limit
  imposed here; callers pass whatever window they searched for the
  first-passage time). This is a pure "was the number right eventually"
  check — it deliberately ignores timing.
* **Horizon success** — target success AND the target was first reached
  within ``horizon_tolerance_pct`` of the *promised* horizon. This is the
  only one of the five that captures "right price, right time" together,
  and it is the one that should gate any claim like "my 4H predictions are
  80% accurate".
* **Trading success** — whether acting on the prediction (entering in the
  predicted direction at the anchor price, exiting at the promised
  horizon) would have been profitable net of an assumed round-trip
  transaction cost. This is the only criterion that accounts for cost of
  execution; it is possible for a prediction to be directionally correct
  and still a trading failure once costs are included.

Worked example matching the brief's own scenario: a 4H prediction of
$100,000 that isn't reached until 3 days (72h) later, with
``horizon_tolerance_pct=0.25`` (i.e. the hit must land within 4h ± 1h to
count): ``target_success=True`` (the price did get there), but
``horizon_success=False`` (72h is nowhere near 4h ± 25%). See
``tests/test_success_criteria.py::test_price_right_but_wildly_late_is_target_success_not_horizon_success``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuccessEvaluation:
    directional_success: bool
    price_success: bool
    target_success: bool | None       # None if never determined (window still open / not searched)
    horizon_success: bool | None      # None if target_success is None
    trading_success: bool | None      # None if trading_cost_pct not provided
    predicted_return: float
    actual_return_at_horizon: float
    time_error_seconds: int | None    # actual_hit_time - promised_target_time, if target was ever hit

    def to_dict(self) -> dict:
        return {
            "directional_success": self.directional_success,
            "price_success": self.price_success,
            "target_success": self.target_success,
            "horizon_success": self.horizon_success,
            "trading_success": self.trading_success,
            "predicted_return": self.predicted_return,
            "actual_return_at_horizon": self.actual_return_at_horizon,
            "time_error_seconds": self.time_error_seconds,
        }


def evaluate_success(
    anchor_price: float,
    predicted_price: float,
    actual_price_at_horizon: float,
    predicted_horizon_seconds: int,
    actual_target_hit_time_seconds: int | None = None,
    anchor_time_seconds: int | None = None,
    price_tolerance_pct: float = 0.01,
    horizon_tolerance_pct: float = 0.25,
    trading_cost_pct: float | None = 0.001,
) -> SuccessEvaluation:
    """Evaluate one prediction against all five independent success criteria.

    Args:
        anchor_price:                Price when the prediction was made.
        predicted_price:              The model's point prediction.
        actual_price_at_horizon:      The real price observed exactly at
                                      ``anchor_time + predicted_horizon_seconds``
                                      — the one fixed evaluation point
                                      directional/price success use.
        predicted_horizon_seconds:    The promised horizon, in seconds.
        actual_target_hit_time_seconds: Unix timestamp of the first candle
                                      that actually reached
                                      ``predicted_price`` (from an
                                      empirical first-passage search, e.g.
                                      ``time_to_target.first_passage_candles``),
                                      or ``None`` if it was never reached
                                      within the search window (censored —
                                      target/horizon success are then
                                      ``None``, not ``False``: "never hit
                                      (yet)" is a different failure from
                                      "hit at the wrong time").
        anchor_time_seconds:          Unix timestamp of the anchor. Required
                                      if ``actual_target_hit_time_seconds``
                                      is given.
        price_tolerance_pct:          Max relative price error, at the
                                      promised horizon, to count as
                                      "price success".
        horizon_tolerance_pct:        Max relative deviation of the actual
                                      hit time from the promised horizon to
                                      still count as "horizon success"
                                      (e.g. 0.25 on a 4h horizon allows a
                                      hit window of 3h-5h).
        trading_cost_pct:             Assumed round-trip transaction cost
                                      (spread + fees), as a fraction of
                                      price. Pass ``None`` to skip
                                      "trading success" (returns ``None``).

    Returns:
        A :class:`SuccessEvaluation` with all five criteria evaluated
        independently.
    """
    if anchor_price == 0:
        raise ValueError("anchor_price must be nonzero")

    predicted_return = (predicted_price - anchor_price) / anchor_price
    actual_return_at_horizon = (actual_price_at_horizon - anchor_price) / anchor_price

    directional_success = (predicted_return > 0 and actual_return_at_horizon > 0) or (
        predicted_return < 0 and actual_return_at_horizon < 0
    )

    price_error_pct = abs(actual_price_at_horizon - predicted_price) / anchor_price
    price_success = price_error_pct <= price_tolerance_pct

    target_success: bool | None = None
    horizon_success: bool | None = None
    time_error_seconds: int | None = None

    if actual_target_hit_time_seconds is not None:
        if anchor_time_seconds is None:
            raise ValueError(
                "anchor_time_seconds is required when actual_target_hit_time_seconds is given"
            )
        target_success = True
        promised_target_time = anchor_time_seconds + predicted_horizon_seconds
        time_error_seconds = actual_target_hit_time_seconds - promised_target_time
        tolerance_seconds = predicted_horizon_seconds * horizon_tolerance_pct
        horizon_success = abs(time_error_seconds) <= tolerance_seconds
    # else: target was never observed to hit within whatever window the
    # caller searched -- leave target_success/horizon_success as None
    # ("not yet / never hit"), which is a DIFFERENT outcome from False and
    # must be reported separately (Phase 5 of the brief: "TARGET NEVER HIT"
    # vs "TARGET HIT LATE" are different failures).

    trading_success: bool | None = None
    if trading_cost_pct is not None:
        direction_sign = 1.0 if predicted_return >= 0 else -1.0
        trade_return = direction_sign * actual_return_at_horizon - trading_cost_pct
        trading_success = trade_return > 0

    return SuccessEvaluation(
        directional_success=directional_success,
        price_success=price_success,
        target_success=target_success,
        horizon_success=horizon_success,
        trading_success=trading_success,
        predicted_return=predicted_return,
        actual_return_at_horizon=actual_return_at_horizon,
        time_error_seconds=time_error_seconds,
    )
