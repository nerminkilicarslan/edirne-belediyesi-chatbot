"""
Structured logging using loguru.

JSON mode is for production/Docker; human-readable for local dev.
Set LOG_JSON=true in .env to enable JSON output.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

from app.config import settings


class _InterceptHandler(logging.Handler):
    """Route stdlib logging into loguru so third-party libs are captured."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    Path("./logs").mkdir(parents=True, exist_ok=True)

    logger.remove()

    if settings.LOG_JSON:
        fmt = (
            '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSS}Z","level":"{level}",'
            '"name":"{name}","function":"{function}","line":{line},'
            '"message":"{message}"}}'
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        )

    logger.add(sys.stdout, format=fmt, level=settings.LOG_LEVEL, colorize=not settings.LOG_JSON)

    logger.add(
        settings.LOG_FILE,
        format=fmt,
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="14 days",
        compression="gz",
        encoding="utf-8",
        colorize=False,
    )

    # Intercept standard library loggers (uvicorn, chromadb, etc.)
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(
        "Logging initialised | level={} | json={} | file={}",
        settings.LOG_LEVEL,
        settings.LOG_JSON,
        settings.LOG_FILE,
    )
