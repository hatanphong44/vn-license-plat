"""Structured logging setup for LPR Runtime.

Following PLAN.md: structured console output with timestamps.
No external dependencies (no Logstash, no ELK).
"""

import logging
import sys
from datetime import UTC, datetime
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class TimestampFormatter(logging.Formatter):
    """Formatter with HH:MM:SS timestamp prefix."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        record.msg = f"{timestamp} [{record.levelname}] {record.msg}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    debug: bool = False,
    use_color: bool = True,
) -> logging.Logger:
    """Setup structured logging for LPR Runtime.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        debug: Enable debug mode
        use_color: Use colored output for terminal

    Returns:
        Configured root logger
    """
    if debug:
        level = "DEBUG"

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("lpr")
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Console handler with timestamp
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Format: {HH:MM:SS} [LEVEL] message
    fmt = "%(message)s"
    formatter = TimestampFormatter(fmt)

    if use_color:
        formatter = ColoredFormatter(fmt, use_color=True)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name (will be prefixed with 'lpr.')

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"lpr.{name}")
    return logging.getLogger("lpr")


# Convenience functions for common logging patterns
def log_startup(logger: logging.Logger, message: str) -> None:
    """Log startup message with INFO level."""
    logger.info(message)


def log_error(logger: logging.Logger, message: str, exc_info: bool = False) -> None:
    """Log error message."""
    logger.error(message, exc_info=exc_info)


def log_debug(logger: logging.Logger, message: str) -> None:
    """Log debug message."""
    logger.debug(message)


def log_plate_detected(logger: logging.Logger, plate: str, confidence: float) -> None:
    """Log plate detection."""
    logger.info(f"New plate detected: {plate}")


def log_plate_collection(logger: logging.Logger, plate: str, frames: int, total: int) -> None:
    """Log plate collection progress."""
    logger.debug(f"Collection progress: plate={plate} frames={frames}/{total}")


def log_best_result(logger: logging.Logger, plate: str, confidence: float) -> None:
    """Log best result selection."""
    logger.info(f"Best result: plate={plate} confidence={confidence:.3f}")


def log_publishing(logger: logging.Logger, plate: str) -> None:
    """Log event publishing."""
    logger.info(f"Publishing event: plate={plate}")


def log_published(logger: logging.Logger, plate: str, status: int) -> None:
    """Log successful publish."""
    logger.info(f"Event published: plate={plate} status={status}")


def log_camera_disconnected(logger: logging.Logger) -> None:
    """Log camera disconnection."""
    logger.error("Camera disconnected")


def log_inference_error(logger: logging.Logger, details: str) -> None:
    """Log inference error."""
    logger.error(f"Inference error: {details}")


def log_publish_failed(logger: logging.Logger, plate: str, details: str) -> None:
    """Log publish failure."""
    logger.error(f"Publish failed: plate={plate} error={details}")
