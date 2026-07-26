"""Pytest configuration — ensures the repo root is on sys.path so that
``from src.xxx import yyy`` works without needing an editable install.
"""

import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
