"""Centralized logging configuration for featcopilot."""

from __future__ import annotations

import logging
import sys

# Create the logger
logger = logging.getLogger("featcopilot")

# Default handler with line number format
_handler = logging.StreamHandler(sys.stderr)
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
_handler.setFormatter(_formatter)

# Only add handler if not already configured
if not logger.handlers:
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional name for the logger. If None, returns the root featcopilot logger.

    Returns:
        A configured logger instance.
    """
    if name:
        # Strip 'featcopilot.' prefix if present to avoid duplication
        if name.startswith("featcopilot."):
            name = name[len("featcopilot.") :]
        return logging.getLogger(f"featcopilot.{name}")
    return logger


def set_level(level: int | str) -> None:
    """Set the logging level.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, "DEBUG", "INFO")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
