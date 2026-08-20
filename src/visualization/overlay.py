"""Visualization - Real-time overlay rendering.

Responsibilities (per PLAN.md):
- Must not block inference loop
- Run in separate thread
"""

import logging
import threading
import queue
from typing import Optional
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
        annotator: Optional[ResultAnnotator] = None,
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
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the overlay renderer."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop,
            name="lpr-overlay",
            daemon=True,
        )
        self._thread.start()
        logger.info("Overlay renderer started")

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
            try:
                self._frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a frame for display (non-blocking).

        Args:
            frame: Frame to display
        """
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def _render_loop(self) -> None:
        """Render loop (runs in separate thread)."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self._running:
            try:
                # Get frame with timeout
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
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
            cv2.imshow(self.window_name, annotated)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                self._running = False
                break

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


def create_overlay_renderer(
    enabled: bool = True,
    display_fps: bool = True,
    window_name: str = "LPR Runtime",
) -> OverlayRenderer | NoOpOverlayRenderer:
    """Factory to create overlay renderer.

    Args:
        enabled: Whether to enable visualization
        display_fps: Show FPS counter
        window_name: Window name

    Returns:
        Renderer instance
    """
    if not enabled:
        return NoOpOverlayRenderer()

    return OverlayRenderer(
        display_fps=display_fps,
        window_name=window_name,
    )
