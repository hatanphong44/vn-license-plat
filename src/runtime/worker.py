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

from src.camera.base import CameraBase
from src.events import EventPublisher, MultiPlateCollector
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
    preview: bool = False  # Enable camera preview window
    save_frames: bool = False  # Save frames with detections


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
        """Initialize runtime worker.

        Args:
            camera: Camera instance
            pipeline: LPR pipeline
            publisher: Event publisher
            config: Worker configuration
            on_frame: Optional callback for each frame
            on_result: Optional callback for each LPR result
            on_error: Optional callback for errors
            overlay: Optional overlay renderer for preview
        """
        self.camera = camera
        self.pipeline = pipeline
        self.publisher = publisher
        self.config = config or WorkerConfig()
        self.on_frame = on_frame
        self.on_result = on_result
        self.on_error = on_error
        self._overlay = overlay

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

        # Track last processed plate by NORMALIZED TEXT (not box_key)
        # This prevents duplicate events for same plate appearing at different positions
        self._last_processed_plate: str | None = None

        # Save frames for debugging
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
        """Stop the runtime worker gracefully.

        Args:
            timeout: Timeout in seconds
        """
        if not self._running:
            return

        logger.info("Stopping LPR runtime worker...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

        # Stop overlay renderer
        if self._overlay is not None:
            self._overlay.stop()

        self._running = False
        logger.info("LPR runtime worker stopped")

    def _run_loop(self) -> None:
        """Main runtime loop."""
        logger.info("=" * 70)
        logger.info("STARTING 24/7 LPR CAMERA WORKER")
        logger.info(f"Camera: {self.camera.source}")
        logger.info(f"Inference FPS: {self.config.inference_fps}")
        logger.info(f"Max frames per plate: {self.config.max_frames}")
        logger.info("=" * 70)

        while not self._stop_event.is_set():

            try:
                # Connect to camera
                if not self.camera.connect():
                    logger.error("Camera connection failed")
                    self._wait_with_check(self.config.reconnect_delay)
                    continue

                logger.info("Camera connected")

                # Run inference loop
                self._inference_loop()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                if self.on_error:
                    self.on_error(e)

            finally:
                if self.camera:
                    self.camera.disconnect()

            # Reconnect delay
            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {self.config.reconnect_delay}s...")
                self._wait_with_check(self.config.reconnect_delay)

        logger.info("Runtime loop ended")

    def _inference_loop(self) -> None:
        """Run inference loop until stop or disconnect."""
        target_interval = 1.0 / max(self.config.inference_fps, 0.1)
        self._last_inference = time.time()
        self._frame_count = 0
        self._last_plate_count = 0

        while not self._stop_event.is_set():
            # Read frame
            frame = self.camera.read()

            if frame is None:
                logger.warning("Frame read failed, reconnecting...")
                break

            # Optional frame callback
            if self.on_frame:
                try:
                    self.on_frame(frame)
                except Exception as e:
                    logger.debug(f"on_frame error: {e}")

            # Throttle inference
            now = time.time()
            if now - self._last_inference < target_interval:
                continue

            self._last_inference = now

            # Run inference
            try:
                results = self.pipeline.process_frame(frame)

                # Process results
                for result in results:
                    # Safety check: result must be an LPRResult with box attribute
                    if not hasattr(result, 'box') or not hasattr(result, 'plate_normalized'):
                        logger.warning(f"Invalid result type: {type(result)}")
                        continue

                    # Add to collection (will aggregate all OCR readings for same box)
                    completed = self._collector.add_detections([result])

                    # Handle completed collections
                    # IMPORTANT: Only process completed collections here
                    # DO NOT log "New plate detected" for individual frame observations
                    for box_key, best_result in completed:
                        self._handle_completed_collection(box_key, best_result)

                    # Optional result callback
                    if self.on_result:
                        try:
                            self.on_result(result)
                        except Exception as e:
                            logger.debug(f"on_result error: {e}")

            except Exception as e:
                logger.error(f"Inference error: {e}")
                if self.on_error:
                    self.on_error(e)
                results = []  # Reset results on error

            # Update overlay if enabled
            if self._overlay is not None:
                self._overlay.update(frame, results, self.config.inference_fps)


    def _handle_completed_collection(self, box_key: str, best_result) -> None:
        """Handle a completed collection after collection phase finishes.

        IMPORTANT: This method implements the NEW LOGIC:
        1. Get best_plate from best_result (already selected by collection)
        2. Compare with _last_processed_plate (NOT box_key!)
        3. If NEW -> log "New plate detected" -> publish -> update _last_processed_plate
        4. If SAME -> don't publish
        5. Always cleanup collection with mark_sent()

        Args:
            box_key: Key identifying this plate position
            best_result: Best captured result from collection (already selected)
        """
        try:
            if not best_result:
                logger.warning(f"No best result for plate at {box_key}")
                self._collector.mark_sent(box_key)
                return

            best_plate = best_result.plate_normalized

            # LOGIC: Compare best_plate with last_processed_plate (NOT box_key!)
            if best_plate == self._last_processed_plate:
                # SAME PLATE - no duplicate event needed
                logger.debug(f"Plate {best_plate} same as last processed, skipping event")
                # Still cleanup collection
                self._collector.mark_sent(box_key)
                return

            # NEW PLATE - this is where we log "New plate detected"
            # This log ONLY appears AFTER collection completes and best result is selected
            logger.info(f"New plate detected: {best_plate}")

            # Publish event
            success = self._send_plate_event(best_result)

            # Only update last_processed_plate AFTER successful publish
            if success:
                self._last_processed_plate = best_plate
                logger.debug(f"Updated last_processed_plate to: {best_plate}")

            # Always cleanup collection
            self._collector.mark_sent(box_key)

        except Exception as e:
            logger.error(f"Error handling completed collection: {e}")
            # Ensure cleanup even on error
            try:
                self._collector.mark_sent(box_key)
            except Exception:
                pass

    def _send_plate_event(self, best_result) -> bool:
        """Send plate event to publisher.

        Args:
            best_result: Best captured result from collection

        Returns:
            True if publish succeeded, False otherwise
        """
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

            logger.info(f"Best result: plate={plate_text} "
                       f"confidence={best_result.confidence:.3f}")
            logger.info(f"Publishing event: plate={plate_text}")

            # Publish
            success = self.publisher.publish(event)

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
        """Wait with periodic stop check.

        Args:
            seconds: Seconds to wait
        """
        end = time.time() + seconds
        while time.time() < end and not self._stop_event.is_set():
            time.sleep(0.1)

    def get_stats(self) -> dict:
        """Get runtime statistics.

        Returns:
            Stats dict
        """
        return {
            "running": self._running,
            "camera": str(self.camera.source),
            "inference_fps": self.config.inference_fps,
        }

    def _get_box_key(self, result: "LPRResult") -> str:
        """Generate a stable key from bounding box coordinates.

        Uses coarse quantization for stability even with small movements.

        Args:
            result: LPRResult with box coordinates

        Returns:
            String key for box tracking
        """
        x1, y1, x2, y2 = result.box
        # Use raw coordinates with coarse quantization (100px grid)
        # This prevents "new plate" spam when plate moves slightly
        q = 100
        return f"{x1//q}_{y1//q}_{x2//q}_{y2//q}"

    def save_frame(self, frame: np.ndarray, prefix: str = "capture") -> str | None:
        """Save a frame to disk for debugging.

        Args:
            frame: Frame to save
            prefix: Filename prefix

        Returns:
            Path to saved file, or None if failed
        """
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
