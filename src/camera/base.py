"""Camera abstraction base.

Responsibilities (per PLAN.md):
- connect, read, reconnect, release, health
- Pipeline must not directly manage cv2.VideoCapture
"""

import abc

import numpy as np


class CameraBase(abc.ABC):
    """Abstract base class for camera implementations."""

    @abc.abstractmethod
    def connect(self) -> bool:
        """Connect to camera.

        Returns:
            True if connected successfully
        """
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from camera and release resources."""
        ...

    @abc.abstractmethod
    def read(self) -> np.ndarray | None:
        """Read a frame from camera.

        Returns:
            Frame as numpy array, or None if read failed
        """
        ...

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if camera is connected and ready."""
        ...

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Perform health check on camera.

        Returns:
            True if camera is healthy
        """
        ...

    @property
    @abc.abstractmethod
    def source(self) -> str:
        """Get camera source identifier."""
        ...

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int] | None:
        """Get camera resolution (width, height) if available."""
        ...
