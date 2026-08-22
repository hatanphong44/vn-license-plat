"""RTSP stream camera implementation."""

import logging
import time
from threading import Event

import cv2
import numpy as np

from .base import CameraBase

logger = logging.getLogger("lpr.camera.rtsp")


class RTSPCamera(CameraBase):
    """RTSP stream camera implementation with robust reconnection."""

    def __init__(
        self,
        url: str,
        buffer_size: int = 1,
        timeout: int = 10,
        reconnect_delay: float = 3.0,
        max_reconnect_attempts: int = 0,  # 0 = infinite
    ):
        """Initialize RTSP camera.

        Args:
            url: RTSP stream URL
            buffer_size: Capture buffer size
            timeout: Connection timeout in seconds
            reconnect_delay: Base delay before reconnecting in seconds
            max_reconnect_attempts: Max reconnect attempts (0 = infinite)
        """
        self._url = url
        self._buffer_size = buffer_size
        self._timeout = timeout
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts
        self._cap: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] | None = None
        self._last_reconnect: float = 0
        self._reconnect_count: int = 0

        # Backoff configuration
        self._backoff_multiplier: float = 1.5
        self._backoff_max: float = 60.0  # Max 60 seconds between retries

        # For graceful shutdown
        self._stop_event = Event()

    @property
    def source(self) -> str:
        return f"rtsp:{self._url[:50]}..."

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution

    def connect(self) -> bool:
        """Connect to RTSP stream."""
        logger.info(f"Connecting to RTSP stream: {self._url[:50]}...")

        # Close existing connection
        if self._cap is not None:
            self._cap.release()

        # Open RTSP stream with optimized settings
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            logger.error("Failed to open RTSP stream")
            self._cap.release()
            self._cap = None
            return False

        # Configure buffer
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        # Set timeout
        self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout * 1000)
        self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout * 1000)

        # Read resolution
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (w, h)

        self._last_reconnect = time.time()
        self._reconnect_count += 1
        logger.info(f"RTSP camera connected: {w}x{h}")
        return True

    def disconnect(self) -> None:
        """Disconnect and release camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("RTSP camera disconnected")

    def read(self) -> np.ndarray | None:
        """Read a frame from camera."""
        if not self.is_connected():
            # Attempt reconnection with backoff
            if self._should_attempt_reconnect():
                self._attempt_reconnect()
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.warning("RTSP frame read failed")
            self._last_reconnect = time.time()
            # Don't immediately return None - let reconnection happen next call
            return None

        return frame

    def is_connected(self) -> bool:
        """Check if camera is connected and ready."""
        if self._cap is None:
            return False

        # Additional check: try to retrieve a frame to verify stream is alive
        if not self._cap.isOpened():
            return False

        return True

    def health_check(self) -> bool:
        """Perform health check."""
        if not self.is_connected():
            return False

        # Try to read a test frame
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False

        return frame.size > 0

    def reconnect(self) -> bool:
        """Force reconnection to RTSP stream."""
        logger.info("Reconnecting to RTSP stream...")
        self.disconnect()
        self._reset_backoff()
        return self.connect()

    def stop(self) -> None:
        """Signal the camera to stop reconnection attempts."""
        self._stop_event.set()
        self.disconnect()

    def _should_attempt_reconnect(self) -> bool:
        """Check if we should attempt reconnection based on backoff."""
        # Check max attempts limit
        if self._max_reconnect_attempts > 0:
            if self._reconnect_count >= self._max_reconnect_attempts:
                return False

        # Check stop event
        if self._stop_event.is_set():
            return False

        # Check backoff delay
        elapsed = time.time() - self._last_reconnect
        current_delay = self._get_current_delay()

        return elapsed >= current_delay

    def _get_current_delay(self) -> float:
        """Calculate current backoff delay."""
        # Exponential backoff with cap
        delay = self._reconnect_delay * (self._backoff_multiplier ** min(self._reconnect_count, 10))
        return min(delay, self._backoff_max)

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with backoff."""
        if self._stop_event.is_set():
            return

        delay = self._get_current_delay()
        logger.info(f"Attempting RTSP reconnect (attempt #{self._reconnect_count + 1}, delay: {delay:.1f}s)...")

        time.sleep(min(delay, 0.5))  # Don't block too long

        if self._stop_event.is_set():
            return

        if self.connect():
            self._reset_backoff()
            logger.info("RTSP reconnection successful")

    def _reset_backoff(self) -> None:
        """Reset backoff state after successful connection."""
        self._reconnect_count = 0
        self._last_reconnect = time.time()
