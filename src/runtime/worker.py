"""Runtime Worker - 24/7 camera worker.

Responsibilities:
- 24/7 loop, lifecycle, recovery, reconnect, graceful shutdown
- Continuous inference at GPU's actual speed
- 3-second result windows with consensus voting
- Deduplication between windows
"""

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.camera.base import CameraBase
from src.events import EventPublisher
from src.observability import get_profiler
from src.pipeline.lpr_pipeline import LPRPipeline

if TYPE_CHECKING:
    from src.domain.models import LPRResult, CapturedPlate
    from src.visualization import OverlayRenderer

logger = logging.getLogger("lpr.runtime.worker")

# Window configuration
RESULT_WINDOW_SECONDS = 3.0


@dataclass
class PlateObservation:
    """A single plate observation from one frame."""
    plate_normalized: str
    plate: str
    confidence: float
    yolo_score: float
    box: list[int]
    ocr_results: list
    is_valid: bool = True  # Whether the plate passed validation
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerConfig:
    """Configuration for runtime worker."""
    reconnect_delay: float = 3.0
    preview: bool = False
    save_frames: bool = False


@dataclass
class WindowResult:
    """Result of a 3-second window."""
    window_id: int
    duration: float
    observations: int
    valid_observations: int
    invalid_observations: int
    unique_plates: list[str]
    candidate_counts: dict[str, int]
    result: str | None
    confidence: float | None
    action: str  # PUBLISH, SKIP_DUPLICATE, NO_CONFIDENT_RESULT


class LPRRuntimeWorker:
    """24/7 LPR runtime worker.

    Manages the camera loop, inference, and event publishing.
    Uses 3-second windows with consensus voting instead of tracking/collection.
    """

    def __init__(
        self,
        camera: CameraBase,
        pipeline: LPRPipeline,
        publisher: EventPublisher,
        config: WorkerConfig | None = None,
        on_frame: Callable | None = None,
        on_result: Callable | None = None,
        on_error: Callable | None = None,
        overlay: "OverlayRenderer | None" = None,
    ):
        """Initialize runtime worker."""
        self.camera = camera
        self.pipeline = pipeline
        self.publisher = publisher
        self.config = config or WorkerConfig()
        self.on_frame = on_frame
        self.on_result = on_result
        self.on_error = on_error
        self._overlay = overlay

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

        # Window management
        self._window_id = 0
        self._window_start: float = 0.0
        self._window_observations: list[PlateObservation] = []
        self._last_published_plate: str | None = None

        # FPS tracking
        self._frame_count = 0
        self._fps_start_time: float = 0.0
        self._current_fps: float = 0.0

        self._save_dir = "captures"
        self._saved_count = 0

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    def start(self) -> None:
        """Start the runtime worker."""
        if self._running:
            logger.warning("Worker already running")
            return

        logger.info("Starting LPR runtime worker...")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="lpr-runtime-worker",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info("LPR runtime worker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the runtime worker gracefully."""
        if not self._running:
            return

        logger.info("Stopping LPR runtime worker...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

        if self._overlay is not None:
            self._overlay.stop()

        self._running = False
        logger.info("LPR runtime worker stopped")

    def _run_loop(self) -> None:
        """Main runtime loop."""
        logger.info("=" * 56)
        logger.info("STARTING 24/7 LPR CAMERA WORKER")
        logger.info(f"Camera: {self.camera.source}")
        logger.info(f"Result window: {RESULT_WINDOW_SECONDS}s")
        logger.info("=" * 56)

        while not self._stop_event.is_set():

            try:
                if not self.camera.connect():
                    logger.error("Camera connection failed")
                    self._wait_with_check(self.config.reconnect_delay)
                    continue

                logger.info("Camera connected")

                self._inference_loop()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                if self.on_error:
                    self.on_error(e)

            finally:
                if self.camera:
                    self.camera.disconnect()

            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {self.config.reconnect_delay}s...")
                self._wait_with_check(self.config.reconnect_delay)

        logger.info("Runtime loop ended")

    def _inference_loop(self) -> None:
        """Run continuous inference loop until stop or disconnect."""
        profiler = get_profiler()
        self._fps_start_time = time.time()
        self._frame_count = 0
        self._window_start = time.time()
        self._window_observations = []

        while not self._stop_event.is_set():
            # Read frame
            frame = self.camera.read()

            if frame is None:
                logger.warning("Frame read failed, reconnecting...")
                profiler.camera_error()
                break

            profiler.camera_read()
            self._frame_count += 1

            # Calculate FPS every second
            now = time.time()
            if now - self._fps_start_time >= 1.0:
                self._current_fps = self._frame_count / (now - self._fps_start_time)
                self._frame_count = 0
                self._fps_start_time = now

            # Optional frame callback
            if self.on_frame:
                try:
                    self.on_frame(frame)
                except Exception as e:
                    logger.debug(f"on_frame error: {e}")

            # NO THROTTLE - Run inference as fast as possible
            profiler.inference_start()

            # Run inference
            results = []
            try:
                inference_start = time.perf_counter()
                results = self.pipeline.process_frame(frame)
                inference_ms = (time.perf_counter() - inference_start) * 1000

                # Record pipeline stats
                plates_found = len(results)
                profiler.pipeline_call(inference_ms, plates_found)

                # Store observations for current window (NO box tracking)
                for result in results:
                    if not hasattr(result, 'box') or not hasattr(result, 'plate_normalized'):
                        continue

                    # Validate plate using postprocessor
                    is_valid = self.pipeline.postprocessor.normalizer.is_valid(result.plate_normalized)

                    obs = PlateObservation(
                        plate_normalized=result.plate_normalized,
                        plate=result.plate,
                        confidence=result.get_confidence(),
                        yolo_score=result.yolo_score,
                        box=result.box,
                        ocr_results=result.ocr_results,
                        is_valid=is_valid,
                        timestamp=now,
                    )
                    self._window_observations.append(obs)

                    if self.on_result:
                        try:
                            self.on_result(result)
                        except Exception as e:
                            logger.debug(f"on_result error: {e}")

            except Exception as e:
                logger.error(f"Inference error: {e}")
                if self.on_error:
                    self.on_error(e)

            # Check window completion (non-blocking)
            self._check_window_completion(profiler)

            # Update overlay if enabled
            if self._overlay is not None:
                self._overlay.update(frame, results, self._current_fps)

            # Check for periodic summary
            if profiler.enabled and profiler.should_print_summary():
                profiler.print_summary(
                    actual_fps=self._current_fps,
                    window_duration=now - self._window_start,
                    observations=len(self._window_observations),
                )

    def _check_window_completion(self, profiler) -> None:
        """Check if 3-second window is complete and finalize if needed."""
        now = time.time()
        elapsed = now - self._window_start

        if elapsed >= RESULT_WINDOW_SECONDS:
            self._finalize_window(profiler)
            # Start new window immediately
            self._window_id += 1
            self._window_start = now
            self._window_observations = []

    def _finalize_window(self, profiler) -> None:
        """Finalize the current 3-second window and potentially publish."""
        window_start_time = self._window_start
        duration = time.time() - window_start_time

        profiler.window_finalized(self._window_id)

        total_observations = len(self._window_observations)
        valid_observations = [obs for obs in self._window_observations if obs.is_valid]
        invalid_observations = total_observations - len(valid_observations)

        if not valid_observations:
            # No valid observations in this window
            window_result = WindowResult(
                window_id=self._window_id,
                duration=duration,
                observations=total_observations,
                valid_observations=0,
                invalid_observations=invalid_observations,
                unique_plates=[],
                candidate_counts={},
                result=None,
                confidence=None,
                action="NO_CONFIDENT_RESULT",
            )
            profiler.log_window_result(window_result, previous_result=self._last_published_plate)
            return

        # Count only valid observations for frequency
        valid_plate_counts = Counter(obs.plate_normalized for obs in valid_observations)

        # Get the most common valid plate
        most_common_plates = valid_plate_counts.most_common()

        if not most_common_plates:
            window_result = WindowResult(
                window_id=self._window_id,
                duration=duration,
                observations=total_observations,
                valid_observations=len(valid_observations),
                invalid_observations=invalid_observations,
                unique_plates=[],
                candidate_counts={},
                result=None,
                confidence=None,
                action="NO_CONFIDENT_RESULT",
            )
            profiler.log_window_result(window_result, previous_result=self._last_published_plate)
            return

        # Check for tie: if top 2 candidates have the same count
        if len(most_common_plates) >= 2:
            first_count = most_common_plates[0][1]
            second_count = most_common_plates[1][1]
            if first_count == second_count:
                # Tie - cannot determine winner
                window_result = WindowResult(
                    window_id=self._window_id,
                    duration=duration,
                    observations=total_observations,
                    valid_observations=len(valid_observations),
                    invalid_observations=invalid_observations,
                    unique_plates=list(valid_plate_counts.keys()),
                    candidate_counts=dict(valid_plate_counts),
                    result=None,
                    confidence=None,
                    action="NO_CONFIDENT_RESULT",
                )
                profiler.log_window_result(window_result, previous_result=self._last_published_plate)
                return

        # Winner is the plate with the highest frequency
        most_common_plate, _ = most_common_plates[0]

        # Get average confidence for the winning plate
        winning_confidences = [
            obs.confidence for obs in valid_observations
            if obs.plate_normalized == most_common_plate
        ]
        avg_confidence = sum(winning_confidences) / len(winning_confidences)

        window_result = WindowResult(
            window_id=self._window_id,
            duration=duration,
            observations=total_observations,
            valid_observations=len(valid_observations),
            invalid_observations=invalid_observations,
            unique_plates=list(valid_plate_counts.keys()),
            candidate_counts=dict(valid_plate_counts),
            result=most_common_plate,
            confidence=avg_confidence,
            action="",  # Will be set below
        )

        # Deduplication: compare with last published
        if most_common_plate == self._last_published_plate:
            window_result.action = "SKIP_DUPLICATE"
            profiler.log_window_result(window_result, previous_result=self._last_published_plate)
        else:
            # Publish new plate
            window_result.action = "PUBLISH"
            success = self._publish_plate(
                most_common_plate, avg_confidence, len(valid_observations), duration, profiler
            )
            if success:
                self._last_published_plate = most_common_plate

            profiler.log_window_result(window_result, previous_result=self._last_published_plate)

    def _publish_plate(self, plate: str, confidence: float, observations: int,
                      window_duration: float, profiler) -> bool:
        """Publish a plate event."""
        try:
            # Find the best observation for this plate
            best_obs = None
            best_conf = 0.0
            for obs in self._window_observations:
                if obs.plate_normalized == plate and obs.confidence > best_conf:
                    best_obs = obs
                    best_conf = obs.confidence

            if best_obs is None:
                return False

            # Create CapturedPlate-like object for publisher
            class CapturedPlateResult:
                def __init__(self, obs):
                    self.plate_normalized = obs.plate_normalized
                    self.plate = obs.plate
                    self.confidence = obs.confidence
                    self.yolo_score = obs.yolo_score
                    self.box = obs.box
                    self.ocr_results = obs.ocr_results
                    self.frames_count = observations

            best_result = CapturedPlateResult(best_obs)

            # Create event
            event = self.publisher.create_event(
                result=best_result,
                camera=str(self.camera.source),
                frames_count=observations,
            )

            # Publish with timing
            publish_start = time.perf_counter()
            success = self.publisher.publish(event)
            publish_ms = (time.perf_counter() - publish_start) * 1000

            profiler.event_published(publish_ms, success)
            profiler.log_event_published(
                plate=plate,
                frames=observations,
                confidence=confidence,
                publish_success=success,
                latency_ms=publish_ms,
                window_duration=window_duration,
            )

            if success:
                logger.info(f"[EVENT] Published: plate={plate} conf={confidence:.3f}")
                return True
            else:
                logger.error(f"[EVENT] Publish failed: plate={plate}")
                return False

        except Exception as e:
            logger.error(f"Error publishing plate event: {e}")
            return False

    def _wait_with_check(self, seconds: float) -> None:
        """Wait with periodic stop check."""
        end = time.time() + seconds
        while time.time() < end and not self._stop_event.is_set():
            time.sleep(0.1)

    def get_stats(self) -> dict:
        """Get runtime statistics."""
        return {
            "running": self._running,
            "camera": str(self.camera.source),
            "current_fps": self._current_fps,
            "window_id": self._window_id,
        }

    def save_frame(self, frame: np.ndarray, prefix: str = "capture") -> str | None:
        """Save a frame to disk for debugging."""
        import os
        try:
            os.makedirs(self._save_dir, exist_ok=True)
            filename = os.path.join(
                self._save_dir,
                f"{prefix}_{self._saved_count:04d}.jpg"
            )
            cv2.imwrite(filename, frame)
            self._saved_count += 1
            logger.debug(f"Saved frame: {filename}")
            return filename
        except Exception as e:
            logger.warning(f"Failed to save frame: {e}")
            return None
