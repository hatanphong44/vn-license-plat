"""Performance profiler for LPR Runtime.

Provides periodic performance summaries without per-frame logging.
Only active when debug mode is enabled via --debug flag.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

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
        self.inference_skipped: int = 0
        self.throttle_ms: float = 0.0

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

        # Collector
        self.collections_started: int = 0
        self.collections_completed: int = 0
        self.collections_timeout: int = 0
        self.avg_collection_size: float = 0.0

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
        """Record skipped inference due to throttle."""
        if not self._enabled:
            return
        self._stats.inference_skipped += 1

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
        """Record a new collection started."""
        if not self._enabled:
            return
        self._stats.collections_started += 1

    def collection_completed(self, size: int, timeout: bool = False) -> None:
        """Record a collection completed."""
        if not self._enabled:
            return
        self._stats.collections_completed += 1
        if timeout:
            self._stats.collections_timeout += 1
        # Update running average
        n = self._stats.collections_completed
        old_avg = self._stats.avg_collection_size
        self._stats.avg_collection_size = old_avg + (size - old_avg) / n

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

    def print_summary(self, configured_fps: float = 5.0) -> None:
        """Print performance summary."""
        if not self._enabled:
            return

        stats = self._stats
        stats.runtime_seconds = time.time() - self._runtime_start
        stats._last_summary_time = time.time()

        # Calculate effective FPS
        total_frames = stats.inference_cycles + stats.inference_skipped
        effective_fps = total_frames / stats.runtime_seconds if stats.runtime_seconds > 0 else 0

        # Find bottleneck
        bottleneck = self._find_bottleneck()

        # Determine slow stages
        slow_stages = []
        if stats.yolo.avg_ms > self.SLOW_YOLO:
            slow_stages.append(f"YOLO={stats.yolo.avg_ms:.0f}ms")
        if stats.ocr_det.avg_ms > self.SLOW_OCR_DET:
            slow_stages.append(f"OCR_det={stats.ocr_det.avg_ms:.0f}ms")
        if stats.ocr_rec.avg_ms > self.SLOW_OCR_REC:
            slow_stages.append(f"OCR_rec={stats.ocr_rec.avg_ms:.0f}ms")
        if stats.pipeline_total_ms / max(1, stats.pipeline_calls) > self.SLOW_PIPELINE:
            slow_stages.append(f"pipeline={stats.pipeline_total_ms / max(1, stats.pipeline_calls):.0f}ms")

        # Print summary
        logger.info("")
        logger.info("=" * 56)
        logger.info(f"  PERFORMANCE SUMMARY | runtime={stats.runtime_seconds:.1f}s")
        logger.info("=" * 56)
        logger.info("")

        # Camera
        logger.info("  Camera")
        logger.info(f"    frames: {stats.camera_frames}")
        if stats.camera_errors > 0:
            logger.info(f"    errors: {stats.camera_errors}  [WARN]")
        logger.info("")

        # Runtime
        logger.info("  Runtime")
        logger.info(f"    configured FPS: {configured_fps}")
        logger.info(f"    effective FPS: {effective_fps:.1f}")
        logger.info(f"    inference cycles: {stats.inference_cycles}")
        if stats.inference_skipped > 0:
            logger.info(f"    skipped (throttle): {stats.inference_skipped}")
        logger.info("")

        # Models
        logger.info("  Models")
        if stats.yolo.calls > 0:
            logger.info(f"    YOLO plate:       {stats.yolo.avg_ms:5.0f} ms avg  max={stats.yolo.max_ms:.0f} ms  ({stats.yolo.calls} calls)")
        if stats.ocr_det.calls > 0:
            logger.info(f"    OCR detection:    {stats.ocr_det.avg_ms:5.0f} ms avg  max={stats.ocr_det.max_ms:.0f} ms  ({stats.ocr_det.calls} calls)")
        if stats.ocr_rec.calls > 0:
            logger.info(f"    OCR recognition:  {stats.ocr_rec.avg_ms:5.0f} ms avg  max={stats.ocr_rec.max_ms:.0f} ms  ({stats.ocr_rec.calls} calls)")
        logger.info("")

        # Pipeline
        if stats.pipeline_calls > 0:
            avg_pipeline = stats.pipeline_total_ms / stats.pipeline_calls
            logger.info("  Pipeline")
            logger.info(f"    avg: {avg_pipeline:.0f} ms  max={stats.pipeline_max_ms:.0f} ms")
            logger.info(f"    plates detected: {stats.plates_detected}")
            logger.info("")

        # Collector
        if stats.collections_started > 0:
            logger.info("  Collector")
            logger.info(f"    active: {stats.collections_started - stats.collections_completed}")
            logger.info(f"    completed: {stats.collections_completed}")
            if stats.collections_timeout > 0:
                logger.info(f"    timeout: {stats.collections_timeout}  [WARN]")
            logger.info("")

        # Events
        if stats.events_published > 0 or stats.events_failed > 0:
            logger.info("  Events")
            logger.info(f"    published: {stats.events_published}")
            if stats.events_failed > 0:
                logger.info(f"    failed: {stats.events_failed}  [WARN]")
            logger.info("")

        # Bottleneck
        if bottleneck:
            logger.info(f"  [BOTTLENECK] {bottleneck}")
        elif slow_stages:
            logger.info(f"  [SLOW] {', '.join(slow_stages)}")

        logger.info("")
        logger.info("=" * 56)

        # Reset delta counters (keep totals for next interval)
        stats.camera_frames = 0
        stats.camera_errors = 0
        stats.inference_skipped = 0

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
                          publish_success: bool, latency_ms: float) -> None:
        """Log a single event publish."""
        if not self._enabled:
            return
        status = "success" if publish_success else "failed"
        logger.debug(f"Event: plate={plate} frames={frames} conf={confidence:.3f} "
                    f"publish={status} latency={latency_ms:.1f}ms")

    def warn_collection_timeout(self, duration: float) -> None:
        """Warn about collection timeout."""
        if not self._enabled:
            return
        logger.warning(f"Collection timeout: duration={duration:.1f}s")

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
