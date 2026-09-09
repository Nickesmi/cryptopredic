"""Tests for src/research/data_provenance.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data_provenance import (
    AcquisitionSpec,
    DatasetProvenance,
    compute_git_commit,
    hash_dataframe,
    load_provenance,
    save_with_provenance,
)


def _df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=dates)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0},
        index=dates,
    )


def test_hash_dataframe_is_deterministic() -> None:
    df = _df()
    assert hash_dataframe(df) == hash_dataframe(df.copy())


def test_hash_dataframe_changes_with_content() -> None:
    df1, df2 = _df(seed=1), _df(seed=2)
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_dataframe_changes_with_columns() -> None:
    df = _df()
    df2 = df.rename(columns={"volume": "vol"})
    assert hash_dataframe(df) != hash_dataframe(df2)


def test_compute_git_commit_returns_a_40_char_hash_in_this_repo() -> None:
    commit = compute_git_commit()
    assert commit is not None
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_compute_git_commit_returns_none_outside_a_repo(tmp_path) -> None:
    assert compute_git_commit(repo_root=tmp_path) is None


def test_dataset_provenance_for_dataframe_captures_range_and_hash() -> None:
    df = _df()
    provenance = DatasetProvenance.for_dataframe(
        df, source="binance_rest_klines", url_template="https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1h"}, symbol="BTCUSDT", timeframe="1H",
    )
    assert provenance.row_count == len(df)
    assert provenance.content_sha256 == hash_dataframe(df)
    assert provenance.start == str(df.index.min())
    assert provenance.end == str(df.index.max())
    assert provenance.git_commit is not None


def test_save_and_load_provenance_roundtrip(tmp_path) -> None:
    df = _df()
    provenance = DatasetProvenance.for_dataframe(
        df, source="test_source", url_template="https://example/test",
        params={}, symbol="BTCUSDT", timeframe="1H",
    )
    data_path = tmp_path / "btc.csv"
    sidecar = save_with_provenance(df, data_path, provenance)

    assert data_path.exists()
    assert sidecar.exists()

    loaded = load_provenance(data_path)
    assert loaded.content_sha256 == provenance.content_sha256
    assert loaded.row_count == provenance.row_count


def test_acquisition_spec_roundtrips_to_json(tmp_path) -> None:
    spec = AcquisitionSpec(
        name="phase6_btc_eth_universe",
        exchange_or_source="Binance",
        endpoint="/api/v3/klines",
        method="GET",
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="1H",
        start="2021-01-01T00:00:00Z",
        end="2024-01-01T00:00:00Z",
        pagination_rule="startTime advances to last candle open_time + 1ms; stop on short page",
        rate_limit_note="0.25s delay between pages, unauthenticated public endpoint",
    )
    path = tmp_path / "spec.json"
    spec.save(path)

    import json
    loaded = json.loads(path.read_text())
    assert loaded["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert loaded["timestamp_convention"] == "candle open time, left-labeled"
