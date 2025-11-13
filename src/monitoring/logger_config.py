"""
Logging configuration for the trading system.
"""
import logging
import sys
from pathlib import Path
import config


def setup_logging():
    """Configure logging for the trading system."""

    # Create logs directory if it doesn't exist
    config.LOGS_DIR.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

