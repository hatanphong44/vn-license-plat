"""Visualization - Real-time overlay rendering.

Responsibilities (per PLAN.md):
- Must not block inference loop
- Run in separate thread
"""

import contextlib
import logging
import os
import queue
import threading

import cv2
import numpy as np

from src.domain.models import LPRResult
from src.visualization.annotator import ResultAnnotator

logger = logging.getLogger("lpr.visualization.overlay")


class OverlayRenderer:
    """Real-time overlay renderer.

    Runs in a separate thread to avoid blocking inference.
    """

    def __init__(
        self,
        annotator: ResultAnnotator | None = None,
        display_fps: bool = True,
        window_name: str = "LPR Runtime",
    ):
        """Initialize overlay renderer.

        Args:
            annotator: Result annotator
            display_fps: Show FPS counter
            window_name: Window name for display
        """
        self.annotator = annotator or ResultAnnotator()
        self.display_fps = display_fps
        self.window_name = window_name

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._results: list[LPRResult] = []
        self._fps: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the overlay renderer."""
        if self._running:
            return

        self._running = True
        try:
            self._thread = threading.Thread(
                target=self._render_loop,
                name="lpr-overlay",
                daemon=True,
            )
            self._thread.start()
            logger.info("Overlay renderer started")
        except Exception as e:
            logger.warning(f"Failed to start overlay renderer: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop the overlay renderer."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        cv2.destroyAllWindows()
        logger.info("Overlay renderer stopped")

    def update(
        self,
        frame: np.ndarray,
        results: list[LPRResult],
        fps: float = 0.0,
    ) -> None:
        """Update the current frame and results.

        Args:
            frame: Current frame
            results: Current LPR results
            fps: Current FPS
        """
        with self._lock:
            self._results = results
            self._fps = fps

            # Put frame in queue (drop if full)
            with contextlib.suppress(queue.Full):
                self._frame_queue.put_nowait(frame.copy())

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a frame for display (non-blocking).

        Args:
            frame: Frame to display
        """
        with contextlib.suppress(queue.Full):
            self._frame_queue.put_nowait(frame.copy())

    def _render_loop(self) -> None:
        """Render loop (runs in separate thread)."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            logger.warning(f"Cannot create window (no display?): {e}")
            logger.info("Preview disabled - running in headless mode")
            self._running = False
            return

        while self._running:
            try:
                # Get frame with timeout
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception:
                continue

            with self._lock:
                results = self._results
                fps = self._fps

            # Annotate
            annotated = frame.copy()

            # Draw results
            if results:
                annotated = self.annotator.draw_results(annotated, results)

            # Draw FPS
            if self.display_fps and fps > 0:
                annotated = self.annotator.draw_fps(annotated, fps)

            # Display
            with contextlib.suppress(Exception):
                cv2.imshow(self.window_name, annotated)

            # Handle key events
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    self._running = False
                    break
            except Exception:
                break

        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()


class NoOpOverlayRenderer:
    """No-op renderer for when visualization is disabled."""

    def __init__(self):
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update(
        self,
        frame: np.ndarray,
        results: list[LPRResult],
        fps: float = 0.0,
    ) -> None:
        pass

    def push_frame(self, frame: np.ndarray) -> None:
        pass


class HeadlessOverlayRenderer:
    """Headless renderer - saves frames with annotations to disk.

    Use this when no display is available (e.g., WSL, SSH).
    """

    def __init__(
        self,
        annotator: ResultAnnotator | None = None,
        save_dir: str = "captures",
        save_interval_seconds: int = 5,
    ):
        """Initialize headless renderer.

        Args:
            annotator: Result annotator
            save_dir: Directory to save captures
            save_interval_seconds: Save a frame every N seconds
        """
        self.annotator = annotator or ResultAnnotator()
        self.save_dir = save_dir
        self.save_interval_seconds = save_interval_seconds
        self._saved_count = 0
        self._last_save_time = 0.0

    def start(self) -> None:
        """Start the renderer."""
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"Headless renderer started - saving to {self.save_dir}/")
        logger.info(f"Saving frame every {self.save_interval_seconds} seconds")

    def stop(self) -> None:
        """Stop the renderer."""
        logger.info(f"Headless renderer stopped - {self._saved_count} frames saved")

    def update(
        self,
        frame: np.ndarray,
        results: list[LPRResult],
        fps: float = 0.0,
    ) -> None:
        """Update with new frame.

        Args:
            frame: Current frame
            results: Current results
            fps: Current FPS
        """
        import time
        current_time = time.time()

        # Always annotate for potential saving
        annotated = frame.copy()

        # Draw results if any
        if results:
            annotated = self.annotator.draw_results(annotated, results)

        # Save frame every N seconds
        if current_time - self._last_save_time >= self.save_interval_seconds:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            detection_info = f"plates_{len(results)}" if results else "no_plate"
            filename = os.path.join(
                self.save_dir,
                f"capture_{timestamp}_{detection_info}.jpg"
            )
            cv2.imwrite(filename, annotated)
            self._saved_count += 1
            self._last_save_time = current_time
            logger.info(f"[{timestamp}] Saved frame #{self._saved_count}: {filename}")

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a frame for display (non-blocking)."""


def create_overlay_renderer(
    enabled: bool = True,
    display_fps: bool = True,
    window_name: str = "LPR Runtime",
    headless: bool = False,
    save_dir: str = "captures",
    save_interval_seconds: int = 5,
) -> OverlayRenderer | NoOpOverlayRenderer | HeadlessOverlayRenderer:
    """Factory to create overlay renderer.

    Args:
        enabled: Whether to enable visualization
        display_fps: Show FPS counter
        window_name: Window name
        headless: Use headless mode (save to disk instead of display)
        save_dir: Directory for saved frames in headless mode
        save_interval_seconds: Save a frame every N seconds

    Returns:
        Renderer instance
    """
    if not enabled:
        return NoOpOverlayRenderer()

    if headless:
        return HeadlessOverlayRenderer(
            save_dir=save_dir,
            save_interval_seconds=save_interval_seconds,
        )

    # Try to create OpenCV-based renderer
    try:
        return OverlayRenderer(
            display_fps=display_fps,
            window_name=window_name,
        )
    except Exception as e:
        logger.warning(f"Cannot create display renderer: {e}")
        logger.info("Falling back to headless mode")
        return HeadlessOverlayRenderer(
            save_dir=save_dir,
            save_interval_seconds=save_interval_seconds,
        )
