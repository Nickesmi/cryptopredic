"""Run one shadow-mode evaluation cycle: score every expired, unevaluated prediction.

Intended to be invoked on a schedule (cron, Celery beat, a systemd timer —
whatever the deployment already uses). Each invocation is idempotent:
predictions that are already evaluated are never re-scored (see
``PredictionStore.due_predictions``, which only returns predictions with
no matching row in ``prediction_evaluations``).

NOT exercised against live data in this repository's development
environment (no outbound network access here). Test against a real
Binance endpoint before relying on this in production.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp

from src.evaluation.prediction_store import PredictionStore
from src.evaluation.shadow import binance_price_lookup, evaluate_due_predictions


async def main() -> None:
    store = PredictionStore()
    async with aiohttp.ClientSession() as session:
        async def price_lookup(symbol: str, timeframe: str, at_time: int):
            return await binance_price_lookup(symbol, timeframe, at_time, session=session)

        result = await evaluate_due_predictions(store, price_lookup)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
