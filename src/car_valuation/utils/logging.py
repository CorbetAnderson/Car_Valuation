# src/my_project/utils/logging.py
from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(
    *,
    level: str | int = "INFO",
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure root logging once and return a logger.

    Args:
        level: "DEBUG"/"INFO"/"WARNING"/... or an int logging level.
        name: Optional logger name. If None, returns root logger.

    Behavior:
        - Avoids double-handlers if called multiple times.
        - Logs to stdout with timestamps and module names.
        - Respects LOG_LEVEL env var if set (e.g. LOG_LEVEL=DEBUG).
    """
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        level = env_level

    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = int(level)

    root = logging.getLogger()
    root.setLevel(level_value)

    # Prevent duplicated logs if setup_logging() is called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level_value)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)

    return logging.getLogger(name) if name else logging.getLogger()
