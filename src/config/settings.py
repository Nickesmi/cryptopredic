"""
Application-wide configuration loaded from environment variables.

All tuneable constants live here so that the rest of the codebase
imports from a single source of truth and no magic strings are
scattered across modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the repository (two levels up from this file: src/config/settings.py)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data" / "raw")))
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(REPO_ROOT / "models" / "saved")))
PREDICTION_DB_PATH: Path = Path(
    os.getenv(
        "PREDICTION_DB_PATH",
        str(REPO_ROOT / "data" / "prediction_evaluation.sqlite3"),
    )
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------

COINGECKO_BASE_URL: str = os.getenv(
    "COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3"
)

# Number of days of history to fetch when training models
HISTORY_DAYS: int = int(os.getenv("HISTORY_DAYS", "365"))

# ---------------------------------------------------------------------------
# Supported assets  (use CoinGecko coin IDs)
# ---------------------------------------------------------------------------

TIER1_SYMBOLS: list[str] = ["bitcoin", "ethereum", "solana"]

# ---------------------------------------------------------------------------
# Forecast horizons (in calendar days)
# ---------------------------------------------------------------------------

FORECAST_HORIZONS: list[int] = [1, 7, 30]

# ---------------------------------------------------------------------------
# Model hyper-parameters (XGBoost defaults — override via env)
# ---------------------------------------------------------------------------

XGB_N_ESTIMATORS: int = int(os.getenv("XGB_N_ESTIMATORS", "500"))
XGB_MAX_DEPTH: int = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_LEARNING_RATE: float = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_SUBSAMPLE: float = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE_BYTREE: float = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))
XGB_RANDOM_STATE: int = int(os.getenv("XGB_RANDOM_STATE", "42"))

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

HTTP_TIMEOUT_SECONDS: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
