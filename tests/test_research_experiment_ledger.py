"""Tests for src/research/experiment_ledger.py."""

from __future__ import annotations

from src.research.experiment_ledger import ExperimentLedger, ExperimentRecord


def _record(i: int) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=f"exp{i}",
        family="momentum",
        signal_name="momentum",
        horizon_label="1D",
        universe="realistic",
        region="validation",
        ic_method="spearman",
        n_periods=100,
        mean_ic=0.02,
        median_ic=0.01,
        std_ic=0.1,
        ic_information_ratio=0.2,
        ci_low=-0.01,
        ci_high=0.05,
        hit_rate=0.55,
        quintile_monotonic=False,
        quintile_spearman=0.3,
        top_minus_bottom=0.01,
    )


def test_add_and_dataframe_roundtrip() -> None:
    ledger = ExperimentLedger()
    for i in range(3):
        ledger.add(_record(i))
    assert len(ledger) == 3
    df = ledger.to_dataframe()
    assert len(df) == 3
    assert "mean_ic" in df.columns


def test_empty_ledger_dataframe() -> None:
    ledger = ExperimentLedger()
    df = ledger.to_dataframe()
    assert df.empty


def test_save_and_load_roundtrip(tmp_path) -> None:
    ledger = ExperimentLedger()
    for i in range(5):
        ledger.add(_record(i))
    path = tmp_path / "ledger.jsonl"
    ledger.save(path)

    loaded = ExperimentLedger.load(path)
    assert len(loaded) == 5
    assert loaded.records[0].experiment_id == "exp0"


def test_load_missing_file_returns_empty_ledger(tmp_path) -> None:
    loaded = ExperimentLedger.load(tmp_path / "does_not_exist.jsonl")
    assert len(loaded) == 0
