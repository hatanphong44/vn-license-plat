"""Video file camera implementation (for testing)."""

import logging
from typing import Optional
import cv2
import numpy as np

from .base import CameraBase


logger = logging.getLogger("lpr.camera.video")


class VideoFileCamera(CameraBase):
    """Video file camera for testing without real camera."""

    def __init__(
        self,
        video_path: str,
        loop: bool = True,
    ):
        """Initialize video file camera.

        Args:
            video_path: Path to video file
            loop: Loop video when it ends
        """
        self._video_path = video_path
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._resolution: tuple[int, int] | None = None
        self._frame_count: int = 0
        self._total_frames: int = 0
        self._fps: float = 0

    @property
    def source(self) -> str:
        return f"video:{self._video_path}"

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution

    @property
    def frame_count(self) -> int:
        """Current frame number."""
        return self._frame_count

    @property
    def total_frames(self) -> int:
        """Total frames in video."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Video FPS."""
        return self._fps

    def connect(self) -> bool:
        """Open video file."""
        logger.info(f"Opening video file: {self._video_path}")

        self._cap = cv2.VideoCapture(self._video_path)

        if not self._cap.isOpened():
            logger.error(f"Failed to open video file: {self._video_path}")
            self._cap.release()
            self._cap = None
            return False

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (w, h)

        self._frame_count = 0
        logger.info(f"Video opened: {w}x{h}, {self._total_frames} frames, {self._fps:.1f} FPS")
        return True

    def disconnect(self) -> None:
        """Close video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video file closed")

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from video."""
        if not self.is_connected():
            return None

        ok, frame = self._cap.read()
        self._frame_count += 1

        if not ok or frame is None:
            if self._loop:
                # Loop back to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_count = 0
                ok, frame = self._cap.read()
                self._frame_count = 1
                return frame if ok else None
            return None

        return frame

    def is_connected(self) -> bool:
        """Check if video is open."""
        return self._cap is not None and self._cap.isOpened()

    def health_check(self) -> bool:
        """Video file is always healthy if connected."""
        return self.is_connected()

    def seek(self, frame_number: int) -> bool:
        """Seek to specific frame."""
        if not self.is_connected():
            return False
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._frame_count = frame_number
        return True
