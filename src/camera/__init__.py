"""Camera package.

Factory functions to create camera instances based on source type.
"""

import logging

from .base import CameraBase
from .rtsp import RTSPCamera
from .usb import USBCamera

logger = logging.getLogger("lpr.camera")


def create_camera(
    source: str | int,
    buffer_size: int = 1,
    timeout: int = 10,
    reconnect_delay: float = 3.0,
) -> CameraBase:
    """Create appropriate camera instance based on source.

    Args:
        source: Camera source (0, 1 for USB, or RTSP URL)
        buffer_size: Capture buffer size
        timeout: Connection timeout for network cameras
        reconnect_delay: Delay before reconnecting

    Returns:
        Camera instance

    Raises:
        ValueError: If source type is not recognized
    """
    source_str = str(source)

    # Check if it's a number (USB camera)
    try:
        device_id = int(source)
        logger.info(f"Creating USB camera for device {device_id}")
        return USBCamera(
            device_id=device_id,
            buffer_size=buffer_size,
        )
    except ValueError:
        pass

    # Check if it's RTSP URL
    if source_str.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        logger.info("Creating RTSP camera for URL")
        return RTSPCamera(
            url=source_str,
            buffer_size=buffer_size,
            timeout=timeout,
            reconnect_delay=reconnect_delay,
        )

    # Default to RTSP (might be a path without extension)
    logger.info("Creating RTSP camera as default")
    return RTSPCamera(
        url=source_str,
        buffer_size=buffer_size,
        timeout=timeout,
        reconnect_delay=reconnect_delay,
    )


__all__ = [
    "CameraBase",
    "RTSPCamera",
    "USBCamera",
    "create_camera",
]
