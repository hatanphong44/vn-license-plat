"""Logging package."""

from .setup import (
    get_logger,
    log_best_result,
    log_camera_disconnected,
    log_debug,
    log_error,
    log_inference_error,
    log_plate_collection,
    log_plate_detected,
    log_publish_failed,
    log_published,
    log_publishing,
    log_startup,
    setup_logging,
)

__all__ = [
    "get_logger",
    "log_best_result",
    "log_camera_disconnected",
    "log_debug",
    "log_error",
    "log_inference_error",
    "log_plate_collection",
    "log_plate_detected",
    "log_publish_failed",
    "log_published",
    "log_publishing",
    "log_startup",
    "setup_logging",
]
