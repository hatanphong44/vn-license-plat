"""Observability - Metrics collection."""

import logging
import threading
import time
from dataclasses import dataclass

from src.domain.models import LPRResult

logger = logging.getLogger("lpr.observability.metrics")


@dataclass
class RuntimeMetrics:
    """Runtime metrics."""

    # Counters
    frames_processed: int = 0
    plates_detected: int = 0
    events_published: int = 0
    events_failed: int = 0

    # Timing
    total_inference_time: float = 0.0
    avg_inference_time: float = 0.0
    last_inference_time: float = 0.0

    # FPS
    current_fps: float = 0.0
    avg_fps: float = 0.0

    # Errors
    camera_errors: int = 0
    inference_errors: int = 0
    publish_errors: int = 0

    # Plate tracking
    unique_plates: int = 0
    last_plate: str | None = None

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "frames_processed": self.frames_processed,
            "plates_detected": self.plates_detected,
            "events_published": self.events_published,
            "events_failed": self.events_failed,
            "avg_inference_time_ms": self.avg_inference_time * 1000,
            "current_fps": self.current_fps,
            "avg_fps": self.avg_fps,
            "camera_errors": self.camera_errors,
            "inference_errors": self.inference_errors,
            "publish_errors": self.publish_errors,
            "unique_plates": self.unique_plates,
            "last_plate": self.last_plate,
        }


class MetricsCollector:
    """Collect and track runtime metrics."""

    def __init__(self):
        self._metrics = RuntimeMetrics()
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._fps_frames = 0
        self._fps_last_time = time.time()
        self._unique_plates_seen: set = set()

    def record_frame(self, inference_time: float) -> None:
        """Record a processed frame."""
        with self._lock:
            self._metrics.frames_processed += 1
            self._metrics.total_inference_time += inference_time
            self._metrics.last_inference_time = inference_time

            # Calculate averages
            if self._metrics.frames_processed > 0:
                self._metrics.avg_inference_time = (
                    self._metrics.total_inference_time / self._metrics.frames_processed
                )

            # Calculate FPS
            self._fps_frames += 1
            now = time.time()
            elapsed = now - self._fps_last_time

            if elapsed >= 1.0:
                self._metrics.current_fps = self._fps_frames / elapsed
                self._metrics.avg_fps = (
                    self._metrics.frames_processed / (now - self._start_time)
                )
                self._fps_frames = 0
                self._fps_last_time = now

    def record_plate(self, result: LPRResult) -> None:
        """Record a detected plate."""
        with self._lock:
            self._metrics.plates_detected += 1
            self._metrics.last_plate = result.plate_normalized

            if result.plate_normalized not in self._unique_plates_seen:
                self._unique_plates_seen.add(result.plate_normalized)
                self._metrics.unique_plates = len(self._unique_plates_seen)

    def record_event(self, success: bool) -> None:
        """Record an event publish attempt."""
        with self._lock:
            if success:
                self._metrics.events_published += 1
            else:
                self._metrics.events_failed += 1

    def record_camera_error(self) -> None:
        """Record a camera error."""
        with self._lock:
            self._metrics.camera_errors += 1

    def record_inference_error(self) -> None:
        """Record an inference error."""
        with self._lock:
            self._metrics.inference_errors += 1

    def record_publish_error(self) -> None:
        """Record a publish error."""
        with self._lock:
            self._metrics.publish_errors += 1

    def get_metrics(self) -> dict:
        """Get current metrics."""
        with self._lock:
            return self._metrics.to_dict()

    def get_runtime_metrics(self) -> RuntimeMetrics:
        """Get RuntimeMetrics object."""
        with self._lock:
            return self._metrics


# Global metrics collector
_metrics: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
