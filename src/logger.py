"""
Module for setting up a customized logger using the loguru library.

This module provides a function to configure and return a logger instance
with specific formatting, log file settings, and rotation/compression options.

Log files are stored in the 'logs' directory, organized by date, and named with a timestamp.
Each log file is rotated when it reaches 500 MB in size and compressed in ZIP format.

Example usage:
    from logger_setup import setup_logger
    logger = setup_logger()
    logger.info("This is an info message.")
"""
from pathlib import Path

import loguru
from loguru._logger import Logger

from src.config import load_config


def setup_logger() -> Logger:
    """
    Configures and sets up the logger with a custom format and log file settings.

    Returns:
        Logger: The configured logger instance.
    """
    config = load_config()
    log_dir: Path = config.path.log_dir

    formatter = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
    log_file: str = log_dir / "{time:YYYY-MM-DD}/{time:YYYYMMDD_HHmmss}.log"
    loguru.logger.add(log_file, format=formatter, rotation="500 MB", compression="zip")

    return loguru.logger


if __name__ == "__main__":
    logger = setup_logger()
    logger.debug("This is a debug message.")

    logger.info("This is an info message.")

    logger.warning("This is a warning message.")

    logger.error("This is an error message.")

    try:
        raise ValueError("This is a test exception.")
    except ValueError as e:
        logger.exception(f"An exception occurred: {e}")

    metadata = {"key1": "value1", "key2": "value2"}
    logger.info(metadata)

    print("All tests passed.")
