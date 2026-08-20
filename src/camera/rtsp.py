"""RTSP stream camera implementation."""

import logging
import time

import cv2
import numpy as np

from .base import CameraBase

logger = logging.getLogger("lpr.camera.rtsp")


class RTSPCamera(CameraBase):
    """RTSP stream camera implementation."""

    def __init__(
        self,
        url: str,
        buffer_size: int = 1,
        timeout: int = 10,
        reconnect_delay: float = 3.0,
    ):
        """Initialize RTSP camera.

        Args:
            url: RTSP stream URL
            buffer_size: Capture buffer size
            timeout: Connection timeout in seconds
            reconnect_delay: Delay before reconnecting in seconds
        """
        self._url = url
        self._buffer_size = buffer_size
        self._timeout = timeout
        self._reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] | None = None
        self._last_reconnect: float = 0

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
            # Attempt reconnection
            if time.time() - self._last_reconnect > self._reconnect_delay:
                self.connect()
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.warning("RTSP frame read failed")
            self._last_reconnect = time.time()
            return None

        return frame

    def is_connected(self) -> bool:
        """Check if camera is connected and ready."""
        return self._cap is not None and self._cap.isOpened()

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
        time.sleep(self._reconnect_delay)
        return self.connect()
