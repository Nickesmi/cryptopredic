"""Structured logging factory for the Crypto Alpha Engine."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-configured logger.

    Args:
        name:  Module or component name (use ``__name__`` at call site).
        level: Log level; defaults to INFO.

    Returns:
        A :class:`logging.Logger` instance with a StreamHandler attached
        (no duplicate handlers are added on repeated calls).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured — avoid duplicating handlers.
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger.
    logger.propagate = False

    return logger
