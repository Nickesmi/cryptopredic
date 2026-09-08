"""Run a no-leak historical replay backtest from a CSV file."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.evaluation.backtest import BacktestConfig, run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV with time/open/high/low/close/volume columns")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--lookback", type=int, default=120)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce").fillna(
            pd.to_datetime(df["time"], errors="coerce")
        )
        df = df.set_index("time")

    result = run_backtest(
        df,
        BacktestConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon=args.horizon,
            lookback=args.lookback,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
