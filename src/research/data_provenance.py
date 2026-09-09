"""Dataset provenance and a reproducible acquisition specification (Phase 6, Sections 2, 17).

Section 17's reproducibility requirement is explicit: every experiment
must record its data source, data version/hash, git commit, and
parameters, such that the result is reproducible *from the ledger*. This
module is that recording layer, plus the flip side Section 2 asks for
when real data is not currently available: a precise, versioned
specification of exactly what would be fetched, from where, with what
parameters — so a future run in an environment with network access
reproduces the *intended* dataset exactly, not an approximation of it
re-derived from a natural-language description.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def compute_git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash, or None if not in a git repo / git unavailable.

    Never raises: a provenance record with a missing commit hash is
    honest (and should be flagged by a caller that requires one); a
    provenance step that crashes the whole pipeline over `git` being
    absent would not be.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a DataFrame's content (index + values), independent of dtype quirks.

    Uses ``pandas.util.hash_pandas_object`` (row-wise hashes) reduced to a
    single digest, rather than a raw CSV round-trip, so the hash is stable
    across the same logical content regardless of float formatting.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    digest = hashlib.sha256(row_hashes.tobytes())
    digest.update(",".join(map(str, df.columns)).encode())
    return digest.hexdigest()


@dataclass
class DatasetProvenance:
    """Everything needed to know exactly what data an experiment used, and where it came from."""

    source: str  # e.g. "binance_rest_klines", "coingecko_market_chart"
    url_template: str
    params: dict = field(default_factory=dict)
    symbol: str = ""
    timeframe: str = ""
    start: str = ""  # ISO-8601 UTC
    end: str = ""  # ISO-8601 UTC
    retrieved_at: str = ""  # ISO-8601 UTC
    row_count: int = 0
    content_sha256: str = ""
    git_commit: str | None = None
    license_note: str = ""

    @classmethod
    def for_dataframe(
        cls,
        df: pd.DataFrame,
        source: str,
        url_template: str,
        params: dict,
        symbol: str,
        timeframe: str,
        license_note: str = "",
    ) -> "DatasetProvenance":
        start = str(df.index.min()) if not df.empty else ""
        end = str(df.index.max()) if not df.empty else ""
        return cls(
            source=source,
            url_template=url_template,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            row_count=len(df),
            content_sha256=hash_dataframe(df),
            git_commit=compute_git_commit(),
            license_note=license_note,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def save_with_provenance(df: pd.DataFrame, data_path: Path, provenance: DatasetProvenance) -> Path:
    """Write *df* to *data_path* (CSV) and a sidecar ``<data_path>.provenance.json``.

    Returns the provenance sidecar path.
    """
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path)
    provenance_path = data_path.with_suffix(data_path.suffix + ".provenance.json")
    with provenance_path.open("w") as f:
        json.dump(provenance.to_dict(), f, indent=2)
    return provenance_path


def load_provenance(data_path: Path) -> DatasetProvenance:
    provenance_path = data_path.with_suffix(data_path.suffix + ".provenance.json")
    with provenance_path.open() as f:
        payload = json.load(f)
    return DatasetProvenance(**payload)


@dataclass
class AcquisitionSpec:
    """A precise, reproducible specification of a real-data fetch this system would run.

    Serialised to JSON so a future execution (in an environment with the
    network access this sandbox does not have) can be driven by this
    exact file rather than a re-interpretation of prose — the "exact
    reproducible acquisition specification" Phase 6 Section 2 requires
    when the data itself cannot yet be fetched.
    """

    name: str
    exchange_or_source: str
    endpoint: str
    method: str
    symbols: list[str]
    timeframe: str
    start: str  # ISO-8601 UTC
    end: str  # ISO-8601 UTC
    pagination_rule: str
    rate_limit_note: str
    timezone: str = "UTC"
    timestamp_convention: str = "candle open time, left-labeled"
    expected_row_count_per_symbol: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
