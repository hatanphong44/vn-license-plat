"""Runtime Worker - 24/7 camera worker.

Responsibilities (per PLAN.md):
- 24/7 loop, lifecycle, recovery, reconnect, graceful shutdown
"""

import logging
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass

from src.camera.base import CameraBase
from src.pipeline.lpr_pipeline import LPRPipeline
from src.events import MultiPlateCollector, EventPublisher


logger = logging.getLogger("lpr.runtime.worker")


@dataclass
class WorkerConfig:
    """Configuration for runtime worker."""
    inference_fps: float = 5.0
    reconnect_delay: float = 3.0
    max_frames: int = 20
    max_wait_seconds: float = 10.0
    cooldown_seconds: float = 30.0


class LPRRuntimeWorker:
    """24/7 LPR runtime worker.

    Manages the camera loop, inference, and event publishing.
    """

    def __init__(
        self,
        camera: CameraBase,
        pipeline: LPRPipeline,
        publisher: EventPublisher,
        config: Optional[WorkerConfig] = None,
        on_frame: Optional[Callable] = None,
        on_result: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
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
        """
        self.camera = camera
        self.pipeline = pipeline
        self.publisher = publisher
        self.config = config or WorkerConfig()
        self.on_frame = on_frame
        self.on_result = on_result
        self.on_error = on_error

        self._collector = MultiPlateCollector(
            max_frames=self.config.max_frames,
            max_wait_seconds=self.config.max_wait_seconds,
            cooldown_seconds=self.config.cooldown_seconds,
        )

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_inference = 0.0

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
            cap = None

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
                    plate_text = result.plate_normalized

                    # Check if new plate
                    if self._collector.collector.is_new_plate(plate_text):
                        logger.info(f"New plate detected: {plate_text}")

                    # Add to collection
                    completed = self._collector.add_detections([result])

                    # Optional result callback
                    if self.on_result:
                        try:
                            self.on_result(result)
                        except Exception as e:
                            logger.debug(f"on_result error: {e}")

                    # Handle completed collections
                    for plate in completed:
                        self._send_plate_event(plate)

            except Exception as e:
                logger.error(f"Inference error: {e}")
                if self.on_error:
                    self.on_error(e)

    def _send_plate_event(self, plate_text: str) -> None:
        """Send plate event to publisher.

        Args:
            plate_text: Normalized plate text
        """
        try:
            # Get best result
            best = self._collector.get_best_result(plate_text)
            if not best:
                logger.warning(f"No result for plate: {plate_text}")
                return

            # Create event
            event = self.publisher.create_event(
                result=best,
                camera=str(self.camera.source),
                frames_count=self._collector.collector.get_collection(plate_text).size(),
            )

            logger.info(f"Best result: plate={plate_text} "
                       f"confidence={best.confidence:.3f}")
            logger.info(f"Publishing event: plate={plate_text}")

            # Publish
            success = self.publisher.publish(event)

            if success:
                logger.info(f"Event published: plate={plate_text}")
                self._collector.mark_sent(plate_text)
                self._collector.collector.clear(plate_text)
            else:
                logger.error(f"Publish failed: plate={plate_text}")

        except Exception as e:
            logger.error(f"Error sending plate event: {e}")

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
