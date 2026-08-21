"""Tests for src/camera/* modules - full coverage"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestCameraBase:
    """Test CameraBase abstract class."""

    def test_camera_base_is_abstract(self):
        """Test CameraBase cannot be instantiated directly."""
        from src.camera.base import CameraBase

        with pytest.raises(TypeError):
            CameraBase()


class TestUSBCamera:
    """Test USBCamera class."""

    def test_init(self):
        """Test USBCamera initialization."""
        from src.camera.usb import USBCamera

        camera = USBCamera(
            device_id=1,
            width=640,
            height=480,
            buffer_size=2,
        )

        assert camera._device_id == 1
        assert camera._width == 640
        assert camera._height == 480
        assert camera._buffer_size == 2

    def test_init_defaults(self):
        """Test USBCamera with defaults."""
        from src.camera.usb import USBCamera

        camera = USBCamera()
        assert camera._device_id == 0
        assert camera._width is None
        assert camera._height is None
        assert camera._buffer_size == 1

    def test_source_property(self):
        """Test source property."""
        from src.camera.usb import USBCamera

        camera = USBCamera(device_id=2)
        assert camera.source == "usb:2"

    def test_resolution_property_none(self):
        """Test resolution property when not connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()
        assert camera.resolution is None

    def test_connect_fails(self):
        """Test connect when camera cannot be opened."""
        from src.camera.usb import USBCamera

        camera = USBCamera(device_id=99)

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            result = camera.connect()

            assert result is False
            assert camera._cap is None

    def test_connect_success(self):
        """Test successful connect."""
        from src.camera.usb import USBCamera

        camera = USBCamera(device_id=0, width=640, height=480, buffer_size=1)

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            }.get(prop, 0.0)
            mock_cv2.VideoCapture.return_value = mock_cap

            result = camera.connect()

            assert result is True
            assert camera._cap is not None

    def test_disconnect(self):
        """Test disconnect releases camera."""
        from src.camera.usb import USBCamera

        camera = USBCamera()

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            camera.disconnect()

            mock_cap.release.assert_called_once()
            assert camera._cap is None

    def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()
        camera.disconnect()  # Should not raise

    def test_read_when_not_connected(self):
        """Test read returns None when not connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()
        result = camera.read()
        assert result is None

    def test_read_success(self):
        """Test successful read."""
        from src.camera.usb import USBCamera

        camera = USBCamera()

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            frame = camera.read()

            assert frame is not None
            assert frame.shape == (480, 640, 3)

    def test_read_failure(self):
        """Test read returns None on failure."""
        from src.camera.usb import USBCamera

        camera = USBCamera()

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            frame = camera.read()

            assert frame is None

    def test_is_connected_true(self):
        """Test is_connected returns True when connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            assert camera.is_connected() is True

    def test_is_connected_false(self):
        """Test is_connected returns False when not connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()
        assert camera.is_connected() is False

    def test_health_check_connected(self):
        """Test health_check when connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()

        with patch('src.camera.usb.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            assert camera.health_check() is True

    def test_health_check_not_connected(self):
        """Test health_check when not connected."""
        from src.camera.usb import USBCamera

        camera = USBCamera()
        assert camera.health_check() is False


class TestRTSPCamera:
    """Test RTSPCamera class."""

    def test_init(self):
        """Test RTSPCamera initialization."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(
            url="rtsp://192.168.1.100:554/stream",
            buffer_size=2,
            timeout=30,
            reconnect_delay=5.0,
        )

        assert camera._url == "rtsp://192.168.1.100:554/stream"
        assert camera._buffer_size == 2
        assert camera._timeout == 30
        assert camera._reconnect_delay == 5.0

    def test_init_defaults(self):
        """Test RTSPCamera with defaults."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")
        assert camera._buffer_size == 1
        assert camera._timeout == 10
        assert camera._reconnect_delay == 3.0

    def test_source_property(self):
        """Test source property."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://example.com/stream")
        assert "rtsp:" in camera.source

    def test_resolution_property_none(self):
        """Test resolution property when not connected."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")
        assert camera.resolution is None

    def test_connect_fails(self):
        """Test connect when stream cannot be opened."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://invalid.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            result = camera.connect()

            assert result is False
            mock_cap.release.assert_called()

    def test_connect_success(self):
        """Test successful connect."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            }.get(prop, 0.0)
            mock_cv2.VideoCapture.return_value = mock_cap

            result = camera.connect()

            assert result is True
            mock_cap.set.assert_called()

    def test_connect_releases_existing(self):
        """Test connect releases existing connection."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            }.get(prop, 0.0)
            mock_cv2.VideoCapture.return_value = mock_cap

            # First connect
            camera.connect()
            # Second connect
            camera.connect()

            # Should have released the first connection
            mock_cap.release.assert_called()

    def test_disconnect(self):
        """Test disconnect releases camera."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            camera.disconnect()

            mock_cap.release.assert_called_once()
            assert camera._cap is None

    def test_read_when_not_connected(self):
        """Test read returns None when not connected."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        # Force reconnect delay check to pass
        import time
        camera._last_reconnect = time.time() - 10

        result = camera.read()
        assert result is None

    def test_read_success(self):
        """Test successful read."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            frame = camera.read()

            assert frame is not None

    def test_read_failure_triggers_reconnect(self):
        """Test read failure triggers reconnection."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            import time
            camera._last_reconnect = time.time() - 10  # Force reconnect check

            frame = camera.read()

            assert frame is None

    def test_is_connected_true(self):
        """Test is_connected returns True when connected."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            assert camera.is_connected() is True

    def test_is_connected_false(self):
        """Test is_connected returns False when not connected."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")
        assert camera.is_connected() is False

    def test_health_check_connected(self):
        """Test health_check when connected."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap

            camera.connect()
            assert camera.health_check() is True

    def test_health_check_not_connected(self):
        """Test health_check when not connected."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")
        assert camera.health_check() is False

    def test_reconnect(self):
        """Test reconnect method."""
        from src.camera.rtsp import RTSPCamera

        camera = RTSPCamera(url="rtsp://test.com/stream")

        with patch('src.camera.rtsp.cv2') as mock_cv2, patch('time.sleep'):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            }.get(prop, 0.0)
            mock_cv2.VideoCapture.return_value = mock_cap

            result = camera.reconnect()

            assert result is True
