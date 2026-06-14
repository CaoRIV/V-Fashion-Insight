"""Logging helpers for project entry points."""

import logging
from typing import Final

DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
_HANDLER_MARKER: Final[str] = "_v_fashion_insight_handler"


def configure_logging(
    level: int | str = logging.INFO,
    logger_name: str = "v_fashion_insight",
) -> logging.Logger:
    """Configure and return a package logger without changing the root logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    has_project_handler = any(
        getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers
    )
    if not has_project_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced project logger without configuring handlers."""
    return logging.getLogger(f"v_fashion_insight.{name}")
