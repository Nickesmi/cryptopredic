"""Experiment ledger (Phase 5, Section 15).

"Because many signals and horizons will be tested: record EVERY
experiment ... do not discard failed experiments ... create an
experiment ledger so future phases cannot accidentally treat a
discovered pattern as out-of-sample evidence after it has already been
mined." This module is that ledger: an append-only record of every
(signal, horizon, universe, region) cell actually run, persisted to a
JSONL file so it survives this process and can be inspected or extended
by a future phase without re-running Phase 5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_LEDGER_PATH = Path("data/research/phase5_experiment_ledger.jsonl")


@dataclass
class ExperimentRecord:
    experiment_id: str
    family: str
    signal_name: str
    horizon_label: str
    universe: str
    region: str  # "design" | "validation" | "test"
    ic_method: str
    n_periods: int
    mean_ic: float | None
    median_ic: float | None
    std_ic: float | None
    ic_information_ratio: float | None
    ci_low: float | None
    ci_high: float | None
    hit_rate: float | None
    quintile_monotonic: bool | None
    quintile_spearman: float | None
    top_minus_bottom: float | None
    permutation_p_value: float | None = None
    bh_significant: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentLedger:
    records: list[ExperimentRecord] = field(default_factory=list)

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.records])

    def save(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path = DEFAULT_LEDGER_PATH) -> "ExperimentLedger":
        ledger = cls()
        if not path.exists():
            return ledger
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    ledger.add(ExperimentRecord(**json.loads(line)))
        return ledger
