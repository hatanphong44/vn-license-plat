"""Tests for src/logging/setup.py"""

import logging
from unittest.mock import patch


class TestColoredFormatter:
    """Test ColoredFormatter class."""

    def test_colored_formatter_init(self):
        """Test ColoredFormatter initialization."""
        from src.logging.setup import ColoredFormatter

        formatter = ColoredFormatter("%(message)s", use_color=False)
        assert formatter.use_color is False

    def test_colored_formatter_format_no_color(self):
        """Test ColoredFormatter format without color."""
        from src.logging.setup import ColoredFormatter

        formatter = ColoredFormatter("%(message)s", use_color=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "Test message" in result
        assert "\033[" not in result

    def test_colored_formatter_format_with_color_tty(self):
        """Test ColoredFormatter format with color in TTY."""
        from src.logging.setup import ColoredFormatter

        with patch('sys.stdout.isatty', return_value=True):
            formatter = ColoredFormatter("%(message)s", use_color=True)
            assert formatter.use_color is True

    def test_colored_formatter_colors_defined(self):
        """Test ColoredFormatter has correct colors."""
        from src.logging.setup import ColoredFormatter

        formatter = ColoredFormatter("%(message)s", use_color=False)
        assert "DEBUG" in formatter.COLORS
        assert "INFO" in formatter.COLORS
        assert "WARNING" in formatter.COLORS
        assert "ERROR" in formatter.COLORS
        assert "CRITICAL" in formatter.COLORS


class TestTimestampFormatter:
    """Test TimestampFormatter class."""

    def test_timestamp_formatter_format(self):
        """Test TimestampFormatter adds timestamp prefix."""
        from src.logging.setup import TimestampFormatter

        formatter = TimestampFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "Test message" in result


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        from src.logging.setup import setup_logging

        root_logger = setup_logging(level="INFO")

        assert root_logger is not None
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_setup_logging_debug_mode(self):
        """Test setup_logging sets DEBUG when debug=True."""
        from src.logging.setup import setup_logging

        root_logger = setup_logging(level="INFO", debug=True)

        assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_color(self):
        """Test setup_logging with color enabled."""
        from src.logging.setup import setup_logging

        root_logger = setup_logging(level="INFO", use_color=True)

        assert root_logger is not None

    def test_setup_logging_clears_handlers(self):
        """Test setup_logging clears existing handlers."""
        from src.logging.setup import setup_logging

        # Add a handler before
        root = logging.getLogger("lpr")
        root.addHandler(logging.NullHandler())

        setup_logging(level="INFO")

        # Should have cleared old handlers
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "NullHandler" not in handler_types


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_with_name(self):
        """Test get_logger returns named logger."""
        from src.logging.setup import get_logger

        logger = get_logger("test")
        assert "lpr.test" in logger.name

    def test_get_logger_without_name(self):
        """Test get_logger returns root logger without name."""
        from src.logging.setup import get_logger

        logger = get_logger()
        assert logger.name == "lpr"

    def test_get_logger_returns_logger_instance(self):
        """Test get_logger returns actual Logger instance."""
        from src.logging.setup import get_logger

        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestConvenienceFunctions:
    """Test convenience logging functions."""

    def test_log_startup(self):
        """Test log_startup logs info message."""
        from src.logging.setup import log_startup

        mock_logger = logging.getLogger("test_startup")
        mock_logger.setLevel(logging.INFO)

        with patch.object(mock_logger, 'info') as mock_info:
            log_startup(mock_logger, "Application started")
            mock_info.assert_called_once_with("Application started")

    def test_log_error(self):
        """Test log_error logs error message."""
        from src.logging.setup import log_error

        mock_logger = logging.getLogger("test_error")
        mock_logger.setLevel(logging.ERROR)

        with patch.object(mock_logger, 'error') as mock_error:
            log_error(mock_logger, "Error occurred")
            mock_error.assert_called_once()

    def test_log_debug(self):
        """Test log_debug logs debug message."""
        from src.logging.setup import log_debug

        mock_logger = logging.getLogger("test_debug")
        mock_logger.setLevel(logging.DEBUG)

        with patch.object(mock_logger, 'debug') as mock_debug:
            log_debug(mock_logger, "Debug info")
            mock_debug.assert_called_once_with("Debug info")

    def test_log_plate_detected(self):
        """Test log_plate_detected logs plate info."""
        from src.logging.setup import log_plate_detected

        mock_logger = logging.getLogger("test_plate")
        with patch.object(mock_logger, 'info') as mock_info:
            log_plate_detected(mock_logger, "ABC123", 0.95)
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "ABC123" in call_args

    def test_log_plate_collection(self):
        """Test log_plate_collection logs collection progress."""
        from src.logging.setup import log_plate_collection

        mock_logger = logging.getLogger("test_collection")
        with patch.object(mock_logger, 'debug') as mock_debug:
            log_plate_collection(mock_logger, "ABC123", 5, 20)
            mock_debug.assert_called_once()

    def test_log_best_result(self):
        """Test log_best_result logs result info."""
        from src.logging.setup import log_best_result

        mock_logger = logging.getLogger("test_result")
        with patch.object(mock_logger, 'info') as mock_info:
            log_best_result(mock_logger, "ABC123", 0.95)
            mock_info.assert_called_once()

    def test_log_publishing(self):
        """Test log_publishing logs publishing info."""
        from src.logging.setup import log_publishing

        mock_logger = logging.getLogger("test_publishing")
        with patch.object(mock_logger, 'info') as mock_info:
            log_publishing(mock_logger, "ABC123")
            mock_info.assert_called_once()

    def test_log_published(self):
        """Test log_published logs published status."""
        from src.logging.setup import log_published

        mock_logger = logging.getLogger("test_published")
        with patch.object(mock_logger, 'info') as mock_info:
            log_published(mock_logger, "ABC123", 200)
            mock_info.assert_called_once()

    def test_log_camera_disconnected(self):
        """Test log_camera_disconnected logs error."""
        from src.logging.setup import log_camera_disconnected

        mock_logger = logging.getLogger("test_camera")
        with patch.object(mock_logger, 'error') as mock_error:
            log_camera_disconnected(mock_logger)
            mock_error.assert_called_once()

    def test_log_inference_error(self):
        """Test log_inference_error logs error details."""
        from src.logging.setup import log_inference_error

        mock_logger = logging.getLogger("test_inference")
        with patch.object(mock_logger, 'error') as mock_error:
            log_inference_error(mock_logger, "CUDA out of memory")
            mock_error.assert_called_once()

    def test_log_publish_failed(self):
        """Test log_publish_failed logs failure info."""
        from src.logging.setup import log_publish_failed

        mock_logger = logging.getLogger("test_publish_failed")
        with patch.object(mock_logger, 'error') as mock_error:
            log_publish_failed(mock_logger, "ABC123", "Connection refused")
            mock_error.assert_called_once()
