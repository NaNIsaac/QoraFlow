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

    # 1. Config 로드
    ConfigManager.initialize(r"D:\Members\IsaacYong\Dev\CordaX\config.yaml")
    config = ConfigManager.load_config()
    log_dir: Path = Path(config.path.log_dir) # Path 객체 보장

    # 2. 포맷 분리 (핵심: 파일에는 색상 코드를 넣지 않음)
    # Console용 (Color 포함)
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    # File용 (Plain Text)
    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    # 3. 파일 경로 설정 (Rotation이 있으므로 파일명에 초 단위 시간은 불필요할 수 있음)
    # 매번 실행시마다 분리하고 싶다면 유지, 아니라면 'app.log'로 고정하고 rotation에 맡김
    log_file = log_dir / "{time:YYYY-MM-DD}" / "app_{time:HHmmss}.log"

    # 4. 핸들러 초기화 (remove는 최초 1회만 수행됨)
    logger.remove()
    
    # Console Handler
    logger.add(sys.stderr, format=console_fmt, level=level)
    
    # File Handler
    # enqueue=True: 멀티프로세싱/스레드 환경에서 로그 꺠짐 방지 (Async Safe)
    # backtrace=True, diagnose=True: 에러 발생 시 상세 정보
    logger.add(
        log_file, 
        format=file_fmt, 
        rotation="500 MB", 
        compression="zip", 
        level=level,
        enqueue=True, 
        backtrace=True,
        diagnose=True
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
