"""Logging package."""

from .setup import (
    setup_logging,
    get_logger,
    log_startup,
    log_error,
    log_debug,
    log_plate_detected,
    log_plate_collection,
    log_best_result,
    log_publishing,
    log_published,
    log_camera_disconnected,
    log_inference_error,
    log_publish_failed,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "log_startup",
    "log_error",
    "log_debug",
    "log_plate_detected",
    "log_plate_collection",
    "log_best_result",
    "log_publishing",
    "log_published",
    "log_camera_disconnected",
    "log_inference_error",
    "log_publish_failed",
]
