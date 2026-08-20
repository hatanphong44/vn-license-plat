"""Runtime Worker - 24/7 camera worker.

Responsibilities (per PLAN.md):
- 24/7 loop, lifecycle, recovery, reconnect, graceful shutdown
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

# Try to import torch for GPU memory logging
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.camera.base import CameraBase
from src.events import EventPublisher, MultiPlateCollector
from src.observability import get_profiler
from src.pipeline.lpr_pipeline import LPRPipeline

if TYPE_CHECKING:
    from src.domain.models import LPRResult
    from src.visualization import OverlayRenderer

logger = logging.getLogger("lpr.runtime.worker")


@dataclass
class WorkerConfig:
    """Configuration for runtime worker."""
    inference_fps: float = 5.0
    reconnect_delay: float = 3.0
    max_frames: int = 20
    max_wait_seconds: float = 10.0
    cooldown_seconds: float = 30.0
    preview: bool = False
    save_frames: bool = False


class LPRRuntimeWorker:
    """24/7 LPR runtime worker.

    Manages the camera loop, inference, and event publishing.
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
        self._configured_fps = config.inference_fps if config else 5.0

        self._collector = MultiPlateCollector(
            max_frames=self.config.max_frames,
            max_wait_seconds=self.config.max_wait_seconds,
            cooldown_seconds=self.config.cooldown_seconds,
            box_key_func=self._get_box_key,
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_inference = 0.0

        # Track last processed plate by NORMALIZED TEXT
        self._last_processed_plate: str | None = None

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
        logger.info(f"Inference FPS: {self.config.inference_fps}")
        logger.info(f"Max frames per plate: {self.config.max_frames}")
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
        """Run inference loop until stop or disconnect."""
        target_interval = 1.0 / max(self.config.inference_fps, 0.1)
        self._last_inference = time.time()
        profiler = get_profiler()

        while not self._stop_event.is_set():
            # Read frame
            frame = self.camera.read()

            if frame is None:
                logger.warning("Frame read failed, reconnecting...")
                profiler.camera_error()
                break

            profiler.camera_read()

            # Optional frame callback
            if self.on_frame:
                try:
                    self.on_frame(frame)
                except Exception as e:
                    logger.debug(f"on_frame error: {e}")

            # Throttle inference
            now = time.time()
            if now - self._last_inference < target_interval:
                profiler.inference_skipped()
                continue

            self._last_inference = now
            profiler.inference_start()

            # Run inference
            try:
                inference_start = time.perf_counter()
                results = self.pipeline.process_frame(frame)
                inference_ms = (time.perf_counter() - inference_start) * 1000

                # Record pipeline stats
                plates_found = len(results)
                profiler.pipeline_call(inference_ms, plates_found)

                # Process results
                for result in results:
                    if not hasattr(result, 'box') or not hasattr(result, 'plate_normalized'):
                        logger.warning(f"Invalid result type: {type(result)}")
                        continue

                    completed = self._collector.add_detections([result])

                    for box_key, best_result in completed:
                        self._handle_completed_collection(box_key, best_result, profiler)

                    if self.on_result:
                        try:
                            self.on_result(result)
                        except Exception as e:
                            logger.debug(f"on_result error: {e}")

            except Exception as e:
                logger.error(f"Inference error: {e}")
                if self.on_error:
                    self.on_error(e)
                results = []

            # Update overlay if enabled
            if self._overlay is not None:
                self._overlay.update(frame, results, self.config.inference_fps)

            # Check for periodic summary
            if profiler.enabled and profiler.should_print_summary():
                profiler.print_summary(configured_fps=self._configured_fps)

    def _handle_completed_collection(self, box_key: str, best_result, profiler) -> None:
        """Handle a completed collection."""
        try:
            if not best_result:
                self._collector.mark_sent(box_key)
                return

            best_plate = best_result.plate_normalized
            collection_size = best_result.frames_count

            # Check for timeout
            is_timeout = collection_size < self.config.max_frames
            if is_timeout:
                profiler.warn_collection_timeout(self.config.max_wait_seconds)

            profiler.collection_completed(collection_size, timeout=is_timeout)
            profiler.log_collection_completed(collection_size, best_plate, best_result.confidence)

            # Compare with last processed plate
            if best_plate == self._last_processed_plate:
                self._collector.mark_sent(box_key)
                return

            # Publish event
            success = self._send_plate_event(best_result, profiler)

            if success:
                self._last_processed_plate = best_plate

            self._collector.mark_sent(box_key)

        except Exception as e:
            logger.error(f"Error handling completed collection: {e}")
            try:
                self._collector.mark_sent(box_key)
            except Exception:
                pass

    def _send_plate_event(self, best_result, profiler) -> bool:
        """Send plate event to publisher."""
        try:
            if not best_result:
                return False

            plate_text = best_result.plate_normalized

            # Create event
            event = self.publisher.create_event(
                result=best_result,
                camera=str(self.camera.source),
                frames_count=best_result.frames_count if hasattr(best_result, 'frames_count') else 0,
            )

            # Publish with timing
            publish_start = time.perf_counter()
            success = self.publisher.publish(event)
            publish_ms = (time.perf_counter() - publish_start) * 1000

            profiler.event_published(publish_ms, success)
            profiler.log_event_published(
                plate=plate_text,
                frames=best_result.frames_count if hasattr(best_result, 'frames_count') else 0,
                confidence=best_result.confidence,
                publish_success=success,
                latency_ms=publish_ms
            )

            if success:
                logger.info(f"Event published: plate={plate_text}")
                return True
            else:
                logger.error(f"Publish failed: plate={plate_text}")
                return False

        except Exception as e:
            logger.error(f"Error sending plate event: {e}")
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
            "inference_fps": self.config.inference_fps,
        }

    def _get_box_key(self, result: "LPRResult") -> str:
        """Generate a stable key from bounding box coordinates."""
        x1, y1, x2, y2 = result.box
        q = 100
        return f"{x1//q}_{y1//q}_{x2//q}_{y2//q}"

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
