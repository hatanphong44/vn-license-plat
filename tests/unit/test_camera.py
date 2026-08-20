"""Unit tests for camera module."""

import pytest

from src.camera import CameraBase, create_camera
from src.camera.rtsp import RTSPCamera
from src.camera.usb import USBCamera


class TestCreateCamera:
    """Test camera factory."""

    def test_create_usb_camera_with_int(self):
        """Test creating USB camera with integer device ID."""
        camera = create_camera(source=0)
        assert isinstance(camera, USBCamera)

    def test_create_usb_camera_with_string_int(self):
        """Test creating USB camera with string integer."""
        camera = create_camera(source="1")
        assert isinstance(camera, USBCamera)

    def test_create_rtsp_camera(self):
        """Test creating RTSP camera with URL."""
        camera = create_camera(source="rtsp://example.com/stream")
        assert isinstance(camera, RTSPCamera)

    def test_create_rtsp_camera_with_rtsp_prefix(self):
        """Test creating RTSP camera with rtsp:// prefix."""
        camera = create_camera(source="rtsp://192.168.1.1:554/stream")
        assert isinstance(camera, RTSPCamera)

    def test_create_rtsp_camera_with_http_prefix(self):
        """Test creating RTSP camera with http:// prefix."""
        camera = create_camera(source="http://example.com/stream")
        assert isinstance(camera, RTSPCamera)

    def test_create_rtsp_camera_default(self):
        """Test default is RTSP camera."""
        camera = create_camera(source="/path/to/file")
        assert isinstance(camera, RTSPCamera)

    def test_create_camera_with_options(self):
        """Test creating camera with custom options."""
        camera = create_camera(
            source=0,
            buffer_size=5,
            timeout=30,
            reconnect_delay=10.0,
        )
        assert isinstance(camera, USBCamera)


class TestCameraBase:
    """Test CameraBase abstract class."""

    def test_camera_base_is_abstract(self):
        """Test CameraBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CameraBase()


class TestUSBCamera:
    """Test USB camera implementation."""

    def test_usb_camera_init(self):
        """Test USB camera initialization."""
        camera = USBCamera(device_id=0, buffer_size=2)
        assert camera._device_id == 0
        assert camera.source == "usb:0"

    def test_usb_camera_connect(self):
        """Test USB camera connect."""
        camera = USBCamera(device_id=0)
        # Without actual camera, should return False or raise
        # This tests the method exists and can be called
        camera.connect()
        # May be True or False depending on system

    def test_usb_camera_disconnect(self):
        """Test USB camera disconnect."""
        camera = USBCamera(device_id=0)
        camera.disconnect()  # Should not raise

    def test_usb_camera_read(self):
        """Test USB camera read."""
        camera = USBCamera(device_id=0)
        camera.read()
        # May return None if no camera

    def test_usb_camera_is_connected(self):
        """Test USB camera connection check."""
        camera = USBCamera(device_id=0)
        result = camera.is_connected()
        assert isinstance(result, bool)

    def test_usb_camera_health_check(self):
        """Test USB camera health check."""
        camera = USBCamera(device_id=0)
        result = camera.health_check()
        assert isinstance(result, bool)

    def test_usb_camera_resolution(self):
        """Test USB camera resolution property."""
        camera = USBCamera(device_id=0)
        # Check that resolution property is accessible
        _ = camera.resolution


class TestRTSPCamera:
    """Test RTSP camera implementation."""

    def test_rtsp_camera_init(self):
        """Test RTSP camera initialization."""
        camera = RTSPCamera(
            url="rtsp://example.com/stream",
            buffer_size=3,
            timeout=20,
            reconnect_delay=5.0,
        )
        assert camera._url == "rtsp://example.com/stream"
        assert camera._buffer_size == 3

    def test_rtsp_camera_connect(self):
        """Test RTSP camera connect."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        # Without actual camera, should return False or raise
        camera.connect()

    def test_rtsp_camera_disconnect(self):
        """Test RTSP camera disconnect."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        camera.disconnect()  # Should not raise

    def test_rtsp_camera_read(self):
        """Test RTSP camera read."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        camera.read()
        # May return None if not connected

    def test_rtsp_camera_is_connected(self):
        """Test RTSP camera connection check."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        result = camera.is_connected()
        assert isinstance(result, bool)

    def test_rtsp_camera_health_check(self):
        """Test RTSP camera health check."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        result = camera.health_check()
        assert isinstance(result, bool)

    def test_rtsp_camera_resolution(self):
        """Test RTSP camera resolution property."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        # Check that resolution property is accessible
        _ = camera.resolution

    def test_rtsp_camera_with_custom_buffer_size(self):
        """Test RTSP camera with custom buffer size."""
        camera = RTSPCamera(
            url="rtsp://example.com/stream",
            buffer_size=10,
        )
        assert camera._buffer_size == 10

    def test_rtsp_camera_source_property(self):
        """Test RTSP camera source property."""
        camera = RTSPCamera(url="rtsp://example.com/stream")
        # Source property includes truncated URL
        assert "rtsp:" in camera.source
