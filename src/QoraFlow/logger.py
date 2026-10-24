import sys
from pathlib import Path

from loguru import logger
from loguru._logger import Logger

from .config import ConfigManager

_IS_CONFIGURED = False


def setup_logger(level="INFO") -> Logger:
    """
    Configures the logger. This should be called ONLY ONCE at the application startup.
    """
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return logger

    # Falls back to config.yaml in the working directory when not initialized
    if ConfigManager._config_file is None:
        ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    log_dir: Path = Path(config.path.log_dir)

    # Separate formats: the file sink must not contain colour markup
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    file_fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

    log_file = log_dir / "{time:YYYY-MM-DD}" / "app_{time:HHmmss}.log"

    # Drop loguru's default stderr handler
    logger.remove()

    logger.add(sys.stderr, format=console_fmt, level=level)

    # enqueue=True: avoids interleaved log lines under multiprocessing and threads
    logger.add(
        log_file,
        format=file_fmt,
        rotation="500 MB",
        compression="zip",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    _IS_CONFIGURED = True
    return logger


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
