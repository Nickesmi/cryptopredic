"""Phase 6 real-data acquisition driver.

Implements, end to end, the exact acquisition specification Phase 6
Section 2 requires when real data cannot currently be fetched: what
would be fetched, from where, with what parameters, and with what
provenance -- runnable the moment network access to Binance is
available in this environment.

Usage::

    python scripts/fetch_real_historical_data.py --dry-run
        # Builds and prints/saves the AcquisitionSpec without any network
        # call -- always safe to run, proves the spec itself is valid.

    python scripts/fetch_real_historical_data.py
        # Attempts the real fetch. In this sandbox this currently raises
        # a clear error (outbound HTTPS to api.binance.com is denied by
        # sandbox/org policy -- see docs/PHASE6_REAL_DATA_VALIDATION_REPORT.md
        # Section 1) rather than silently falling back to anything
        # synthetic.

Symbols, range, and timeframe below are the same universe shape Phase
4/5 used (BTC, ETH, a handful of large liquid alts) so a real run can be
compared apples-to-apples against the synthetic-universe reports once
data is available.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp

from src.data.exchange.binance import BinanceAdapter
from src.research.data_provenance import AcquisitionSpec, DatasetProvenance, save_with_provenance
from src.utils.candles import bars_to_frame

OUT_DIR = Path("data/raw/phase6_real_data")  # fetched OHLCV: gitignored (data/raw/), large real market data
SPEC_DIR = Path("data/research")  # the spec itself: small, tracked, part of Phase 6's committed deliverables

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]
TIMEFRAME = "1H"
START = "2021-01-01T00:00:00Z"
END = "2024-01-01T00:00:00Z"


def build_acquisition_spec() -> AcquisitionSpec:
    return AcquisitionSpec(
        name="phase6_binance_hourly_universe",
        exchange_or_source="Binance",
        endpoint="/api/v3/klines",
        method="GET",
        symbols=SYMBOLS,
        timeframe=TIMEFRAME,
        start=START,
        end=END,
        pagination_rule=(
            "startTime begins at range start; after each <=1000-row page, "
            "startTime = last_returned_candle.open_time + 1ms; stop when a "
            "page returns fewer than 1000 rows or startTime > end."
        ),
        rate_limit_note="0.25s delay between pages per symbol; unauthenticated public REST endpoint.",
        expected_row_count_per_symbol=int((_iso_to_ms(END) - _iso_to_ms(START)) / 3_600_000),
    )


def _iso_to_ms(iso: str) -> int:
    import pandas as pd

    return int(pd.Timestamp(iso).timestamp() * 1000)


async def _fetch_one_symbol(adapter: BinanceAdapter, symbol: str, start_ms: int, end_ms: int):
    bars = await adapter.get_historical_klines(symbol, TIMEFRAME, start_ms, end_ms)
    return symbol, bars_to_frame(bars, drop_unclosed=True)


async def run_fetch() -> None:
    spec = build_acquisition_spec()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec.save(SPEC_DIR / "phase6_acquisition_spec.json")

    start_ms, end_ms = _iso_to_ms(spec.start), _iso_to_ms(spec.end)

    async with aiohttp.ClientSession() as session:
        adapter = BinanceAdapter(session=session)
        for symbol in spec.symbols:
            try:
                symbol, df = await _fetch_one_symbol(adapter, symbol, start_ms, end_ms)
            except aiohttp.ClientError as exc:
                print(
                    f"FAILED to fetch {symbol}: {exc}\n"
                    "This is expected in the current sandbox -- outbound HTTPS to "
                    "api.binance.com is denied by org/network policy. Run this script "
                    "in an environment with real network access to Binance to complete "
                    "the Phase 6 real-data acquisition. See "
                    "docs/PHASE6_REAL_DATA_VALIDATION_REPORT.md Section 1.",
                    file=sys.stderr,
                )
                raise

            provenance = DatasetProvenance.for_dataframe(
                df, source="binance_rest_klines",
                url_template="https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": "1h", "startTime": start_ms, "endTime": end_ms},
                symbol=symbol, timeframe=spec.timeframe,
                license_note="Binance public market data API -- see Binance API Terms of Use.",
            )
            path = save_with_provenance(df, OUT_DIR / f"{symbol}.csv", provenance)
            print(f"Fetched {len(df)} rows for {symbol} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build and save the acquisition spec only; no network call.")
    args = parser.parse_args()

    if args.dry_run:
        spec = build_acquisition_spec()
        SPEC_DIR.mkdir(parents=True, exist_ok=True)
        path = SPEC_DIR / "phase6_acquisition_spec.json"
        spec.save(path)
        print(f"Dry run: acquisition spec written to {path}")
        print(spec.to_dict())
        return

    asyncio.run(run_fetch())


if __name__ == "__main__":
    main()
