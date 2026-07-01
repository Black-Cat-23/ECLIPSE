"""Loguru-based structured logging for ECLIPSE."""
import sys
import os
from pathlib import Path
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/eclipse.log",
    json_logs: bool = False
) -> None:
    """
    Configure loguru for ECLIPSE.

    - Console: colored pretty-print
    - File: rotating (10 MB), retained 7 days
    - JSON mode: structured logs for API/production
    """
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if json_logs:
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True
        )
    else:
        logger.add(
            sys.stdout,
            format=log_format,
            level=log_level,
            colorize=True
        )

    # File handler with rotation
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level=log_level,
        format=log_format,
        compression="zip"
    )

    logger.info(f"ECLIPSE logging initialized: level={log_level}, file={log_file}")


# Initialize with env vars on import
setup_logging(
    log_level=os.getenv("ECLIPSE_LOG_LEVEL", "INFO"),
    log_file=os.getenv("ECLIPSE_LOG_FILE", "logs/eclipse.log")
)
