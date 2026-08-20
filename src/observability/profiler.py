"""Performance profiler for LPR Runtime.

Provides periodic performance summaries without per-frame logging.
Only active when debug mode is enabled via --debug flag.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("lpr.profiler")


@dataclass
class ModelStats:
    """Statistics for a model/stage."""
    name: str
    calls: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=20))

    def record(self, duration_ms: float) -> None:
        self.calls += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.recent.append(duration_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls > 0 else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_vals = sorted(self.recent)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class RuntimeStats:
    """Runtime statistics tracker."""

    def __init__(self):
        self.start_time: float = time.time()
        self.runtime_seconds: float = 0.0

        # Camera
        self.camera_frames: int = 0
        self.camera_errors: int = 0

        # Inference
        self.inference_cycles: int = 0

        # Models
        self.yolo = ModelStats("YOLO plate detector")
        self.ocr_det = ModelStats("PP-OCR detection")
        self.ocr_rec = ModelStats("PP-OCR recognition")

        # Pipeline
        self.pipeline_calls: int = 0
        self.pipeline_total_ms: float = 0.0
        self.pipeline_max_ms: float = 0.0

        # Results
        self.plates_detected: int = 0
        self.ocr_results: int = 0

        # Windows
        self.windows_completed: int = 0
        self.windows_with_result: int = 0
        self.windows_published: int = 0
        self.windows_skipped: int = 0

        # Events
        self.events_published: int = 0
        self.events_failed: int = 0
        self.publish_latency_ms: float = 0.0

        # Bottleneck tracking
        self._last_summary_time: float = time.time()
        self._bottleneck: str = ""


class DebugProfiler:
    """Periodic performance profiler for LPR Runtime.

    Collects stats and prints summary every N seconds.
    Does NOT log per-frame - only periodic summaries.
    """

    # Summary interval in seconds
    SUMMARY_INTERVAL: float = 5.0

    # Slow thresholds (ms)
    SLOW_YOLO: float = 200.0
    SLOW_OCR_DET: float = 200.0
    SLOW_OCR_REC: float = 500.0
    SLOW_PIPELINE: float = 1000.0

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._stats = RuntimeStats()
        self._runtime_start: float = 0.0
        self._current_inference_start: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    # --- Timing methods ---

    def runtime_start(self) -> None:
        """Mark the start of the runtime."""
        if not self._enabled:
            return
        self._runtime_start = time.time()
        self._stats.start_time = time.time()

    def camera_read(self) -> None:
        """Record a camera frame read."""
        if not self._enabled:
            return
        self._stats.camera_frames += 1

    def camera_error(self) -> None:
        """Record a camera read error."""
        if not self._enabled:
            return
        self._stats.camera_errors += 1

    def inference_start(self) -> None:
        """Mark start of inference cycle."""
        if not self._enabled:
            return
        self._current_inference_start = time.time()

    def inference_skipped(self) -> None:
        """Record skipped inference due to throttle (deprecated - no longer used)."""
        # No longer used

    def yolo_call(self, duration_ms: float) -> None:
        """Record YOLO inference time."""
        if not self._enabled:
            return
        self._stats.yolo.record(duration_ms)

    def ocr_detection(self, duration_ms: float) -> None:
        """Record OCR detection time."""
        if not self._enabled:
            return
        self._stats.ocr_det.record(duration_ms)

    def ocr_recognition(self, duration_ms: float) -> None:
        """Record OCR recognition time."""
        if not self._enabled:
            return
        self._stats.ocr_rec.record(duration_ms)

    def pipeline_call(self, duration_ms: float, plates_found: int = 0) -> None:
        """Record pipeline call."""
        if not self._enabled:
            return
        self._stats.inference_cycles += 1
        self._stats.pipeline_calls += 1
        self._stats.pipeline_total_ms += duration_ms
        self._stats.pipeline_max_ms = max(self._stats.pipeline_max_ms, duration_ms)
        self._stats.plates_detected += plates_found

    def ocr_result(self) -> None:
        """Record an OCR result."""
        if not self._enabled:
            return
        self._stats.ocr_results += 1

    def collection_started(self) -> None:
        """Record a new collection started (deprecated - no longer used)."""
        # Deprecated - no longer used

    def collection_completed(self, size: int, timeout: bool = False) -> None:
        """Record a collection completed (deprecated - no longer used)."""
        # Deprecated - no longer used

    def window_finalized(self, window_id: int) -> None:
        """Record a window finalized."""
        if not self._enabled:
            return
        self._stats.windows_completed += 1

    def log_window_result(self, window_result, previous_result: str | None) -> None:
        """Log window result summary."""
        if not self._enabled:
            return

        stats = self._stats
        stats.windows_completed += 1

        if window_result.action == "PUBLISH":
            stats.windows_published += 1
            stats.windows_with_result += 1
        elif window_result.action == "SKIP_DUPLICATE":
            stats.windows_skipped += 1
            stats.windows_with_result += 1
        # NO_CONFIDENT_RESULT doesn't count as windows_with_result

        # Format: [WINDOW] id=... duration=... observations=... valid_observations=... invalid_observations=... result=... confidence=... previous_result=... action=...
        logger.info("")
        logger.info("[WINDOW]")
        logger.info(f"id={window_result.window_id}")
        logger.info(f"duration={window_result.duration:.2f}s")
        logger.info(f"observations={window_result.observations}")
        logger.info(f"valid_observations={window_result.valid_observations}")
        logger.info(f"invalid_observations={window_result.invalid_observations}")

        # Log candidates with counts
        if window_result.candidate_counts:
            logger.info("candidates:")
            for plate, count in sorted(window_result.candidate_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {plate}: count={count}")

        if window_result.result:
            logger.info(f"result={window_result.result}")
            if window_result.confidence:
                logger.info(f"confidence={window_result.confidence:.3f}")
        else:
            logger.info("result=None")

        logger.info(f"previous_result={previous_result or 'None'}")
        logger.info(f"action={window_result.action}")

    def event_published(self, latency_ms: float, success: bool) -> None:
        """Record an event publish."""
        if not self._enabled:
            return
        if success:
            self._stats.events_published += 1
        else:
            self._stats.events_failed += 1
        self._stats.publish_latency_ms = latency_ms

    def should_print_summary(self) -> bool:
        """Check if it's time to print a summary."""
        if not self._enabled:
            return False
        elapsed = time.time() - self._stats._last_summary_time
        return elapsed >= self.SUMMARY_INTERVAL

    def print_summary(self, actual_fps: float = 0.0, window_duration: float = 0.0,
                     observations: int = 0) -> None:
        """Print performance summary."""
        if not self._enabled:
            return

        stats = self._stats
        stats.runtime_seconds = time.time() - self._runtime_start
        stats._last_summary_time = time.time()

        # Calculate averages
        yolo_ms = stats.yolo.avg_ms if stats.yolo.calls > 0 else 0.0
        ocr_det_ms = stats.ocr_det.avg_ms if stats.ocr_det.calls > 0 else 0.0
        ocr_rec_ms = stats.ocr_rec.avg_ms if stats.ocr_rec.calls > 0 else 0.0
        pipeline_ms = stats.pipeline_total_ms / max(1, stats.pipeline_calls)

        # Format: [PERF] key=value key=value ...
        logger.info("")
        logger.info("[PERF]")
        logger.info(f"actual_inference_fps={actual_fps:.1f}")
        logger.info(f"inference_cycles={stats.inference_cycles}")
        logger.info(f"window_elapsed={window_duration:.2f}s")
        logger.info(f"observations={observations}")
        logger.info(f"yolo_ms={yolo_ms:.0f}")
        logger.info(f"ocr_det_ms={ocr_det_ms:.0f}")
        logger.info(f"ocr_rec_ms={ocr_rec_ms:.0f}")
        logger.info(f"pipeline_ms={pipeline_ms:.0f}")

        # Reset delta counters (keep totals for next interval)
        stats.camera_frames = 0
        stats.camera_errors = 0

    def _find_bottleneck(self) -> str:
        """Identify the main bottleneck."""
        stats = self._stats

        times = []
        if stats.yolo.avg_ms > 0:
            times.append(("YOLO", stats.yolo.avg_ms))
        if stats.ocr_det.avg_ms > 0:
            times.append(("OCR detection", stats.ocr_det.avg_ms))
        if stats.ocr_rec.avg_ms > 0:
            times.append(("OCR recognition", stats.ocr_rec.avg_ms))

        if not times:
            return ""

        # Sort by time descending
        times.sort(key=lambda x: x[1], reverse=True)
        name, ms = times[0]

        return f"{name} (~{ms:.0f} ms)"

    def log_collection_completed(self, observations: int, plate: str, confidence: float) -> None:
        """Log a single collection completion."""
        if not self._enabled:
            return
        logger.debug(f"Collection completed: obs={observations} plate={plate} conf={confidence:.3f}")

    def log_event_published(self, plate: str, frames: int, confidence: float,
                          publish_success: bool, latency_ms: float,
                          window_duration: float = 0.0) -> None:
        """Log a single event publish."""
        if not self._enabled:
            return
        status = "success" if publish_success else "failed"
        logger.info("")
        logger.info("[EVENT]")
        logger.info(f"plate={plate}")
        logger.info(f"confidence={confidence:.3f}")
        logger.info(f"window_duration={window_duration:.2f}s")
        logger.info(f"status={status}")
        logger.info(f"latency_ms={latency_ms:.1f}")

    def warn_collection_timeout(self, duration: float) -> None:
        """Warn about collection timeout (deprecated - no longer used)."""
        # Deprecated - no longer used

    def reset(self) -> None:
        """Reset all stats."""
        self._stats = RuntimeStats()


# Global profiler instance
_profiler: DebugProfiler | None = None


def get_profiler() -> DebugProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = DebugProfiler(enabled=False)
    return _profiler


def init_profiler(enabled: bool = False) -> DebugProfiler:
    """Initialize global profiler."""
    global _profiler
    _profiler = DebugProfiler(enabled=enabled)
    if enabled:
        _profiler.runtime_start()
    return _profiler


def set_profiler_enabled(enabled: bool) -> None:
    """Enable/disable global profiler."""
    global _profiler
    if _profiler is None:
        _profiler = DebugProfiler(enabled=enabled)
    else:
        _profiler._enabled = enabled
