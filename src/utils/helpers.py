"""Shared pure-function helpers used across the Crypto Alpha Engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any missing parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`~pathlib.Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist *df* to a Parquet file, creating parent directories as needed.

    Args:
        df:   DataFrame to serialise.
        path: Destination file path (should end with ``.parquet``).

    Returns:
        The resolved path that was written.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=True)
    return p


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Args:
        path: Path to an existing Parquet file.

    Returns:
        Loaded :class:`~pandas.DataFrame`.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    return pd.read_parquet(p)
