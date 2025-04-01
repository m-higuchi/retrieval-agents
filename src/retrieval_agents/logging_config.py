"""Initializer for logging."""

import logging


def setup_logging() -> None:
    """Set up logging."""
    logger = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return

    file_handler = logging.FileHandler("debug.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s"
        )
    )
    logger.addHandler(file_handler)
