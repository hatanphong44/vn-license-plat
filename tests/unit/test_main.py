"""Tests for src/main.py"""

from unittest.mock import MagicMock, patch

import pytest


class TestMainFunctions:
    """Test main module functions."""

    def test_log_debug_startup_no_debug(self):
        """Test _log_debug_startup returns early when debug is False."""
        from src.main import _log_debug_startup

        mock_logger = MagicMock()
        mock_settings = MagicMock()

        _log_debug_startup(mock_logger, mock_settings, debug=False)

        # Should not log anything when debug is False
        mock_logger.info.assert_not_called()

    def test_log_debug_startup_with_debug(self):
        """Test _log_debug_startup logs camera and model info."""
        from src.main import _log_debug_startup

        mock_logger = MagicMock()
        mock_settings = MagicMock()
        mock_settings.camera.CAMERA_SOURCE = "0"
        mock_settings.camera.CAMERA_BUFFER_SIZE = 1
        mock_settings.models.YOLO_MODEL_PATH = "test.pt"
        mock_settings.models.PADDLE_MODEL_DIR = "./models"
        mock_settings.models.YOLO_DEVICE = "cpu"
        mock_settings.models.PADDLE_DEVICE = "cpu"

        # Mock torch and cv2 imports to raise ImportError
        with patch.dict('sys.modules', {'torch': None}), patch('builtins.__import__', side_effect=ImportError):
            _log_debug_startup(mock_logger, mock_settings, debug=True)

        # Should have logged startup info
        assert mock_logger.info.call_count >= 5


class TestLazyApp:
    """Test lazy app initialization."""

    def test_lazy_app_getattr(self):
        """Test _LazyApp delegates attribute access."""
        from src.main import _LazyApp

        mock_app = MagicMock()
        mock_app.test_attr = "test_value"

        with patch('src.main.get_app', return_value=mock_app):
            lazy = _LazyApp()
            assert lazy.test_attr == "test_value"

    def test_lazy_app_getattr_nonexistent(self):
        """Test _LazyApp raises AttributeError for nonexistent attributes."""
        from src.main import _LazyApp

        mock_app = MagicMock(spec=[])  # Empty mock raises AttributeError

        with patch('src.main.get_app', return_value=mock_app):
            lazy = _LazyApp()
            with pytest.raises(AttributeError):
                _ = lazy.nonexistent
