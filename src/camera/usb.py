"""USB/Webcam camera implementation."""

import logging

import cv2
import numpy as np

from .base import CameraBase

logger = logging.getLogger("lpr.camera.usb")


class USBCamera(CameraBase):
    """USB webcam camera implementation."""

    def __init__(
        self,
        device_id: int = 0,
        width: int | None = None,
        height: int | None = None,
        buffer_size: int = 1,
    ):
        """Initialize USB camera.

        Args:
            device_id: Camera device ID (0, 1, etc.)
            width: Target frame width (None = default)
            height: Target frame height (None = default)
            buffer_size: Capture buffer size
        """
        self._device_id = device_id
        self._width = width
        self._height = height
        self._buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] | None = None

    @property
    def source(self) -> str:
        return f"usb:{self._device_id}"

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution

    def connect(self) -> bool:
        """Connect to USB camera."""
        logger.info(f"Connecting to USB camera {self._device_id}...")

        self._cap = cv2.VideoCapture(self._device_id)

        if not self._cap.isOpened():
            logger.error(f"Failed to open USB camera {self._device_id}")
            self._cap.release()
            self._cap = None
            return False

        # Configure buffer
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        # Set resolution if specified
        if self._width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Read actual resolution
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (w, h)

        logger.info(f"USB camera connected: {w}x{h}")
        return True

    def disconnect(self) -> None:
        """Disconnect and release camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("USB camera disconnected")

    def read(self) -> np.ndarray | None:
        """Read a frame from camera."""
        if not self.is_connected():
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
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

        # Put the frame back (read twice to clear buffer)
        return frame.size > 0
