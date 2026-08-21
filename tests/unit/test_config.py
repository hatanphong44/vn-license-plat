"""Tests for src/config/* modules"""

from unittest.mock import patch


class TestModelSettings:
    """Test ModelSettings class."""

    def test_init_defaults(self):
        """Test ModelSettings with default values."""
        from src.config.settings import ModelSettings

        settings = ModelSettings()

        assert settings.MODEL_BASE_DIR == "./models"
        assert settings.YOLO_MODEL_PATH == "./models/Plate.pt"
        assert settings.YOLO_DEVICE == "0"
        assert settings.YOLO_CONF == 0.25
        assert settings.YOLO_IOU == 0.45
        assert settings.PADDLE_DEVICE == "gpu:0"
        assert settings.OCR_UPSCALE == 4

    def test_init_from_env(self):
        """Test ModelSettings from environment variables."""
        with patch.dict("os.environ", {
            "YOLO_MODEL_PATH": "/custom/path.pt",
            "YOLO_DEVICE": "cpu",
            "YOLO_CONF": "0.5",
        }):
            from src.config.settings import ModelSettings

            settings = ModelSettings()
            assert settings.YOLO_MODEL_PATH == "/custom/path.pt"
            assert settings.YOLO_DEVICE == "cpu"
            assert settings.YOLO_CONF == 0.5


class TestCameraSettings:
    """Test CameraSettings class."""

    def test_init_defaults(self):
        """Test CameraSettings with default values."""
        from src.config.settings import CameraSettings

        settings = CameraSettings()

        assert settings.CAMERA_SOURCE == "0"
        assert settings.INFERENCE_FPS == 5.0
        assert settings.CAMERA_RECONNECT_DELAY == 3
        assert settings.CAMERA_CONNECT_TIMEOUT == 10
        assert settings.CAMERA_BUFFER_SIZE == 1


class TestRuntimeSettings:
    """Test RuntimeSettings class."""

    def test_init_defaults(self):
        """Test RuntimeSettings with default values."""
        from src.config.settings import RuntimeSettings

        settings = RuntimeSettings()

        assert settings.PLATE_COOLDOWN_SECONDS == 30.0
        assert settings.MAX_CAPTURE_FRAMES == 20
        assert settings.MAX_CAPTURE_WAIT_SECONDS == 10.0


class TestEventSettings:
    """Test EventSettings class."""

    def test_init_defaults(self):
        """Test EventSettings with default values."""
        from src.config.settings import EventSettings

        settings = EventSettings()

        assert settings.CALLBACK_URL == ""
        assert settings.CALLBACK_TIMEOUT == 5.0
        assert settings.CALLBACK_RETRY_COUNT == 3
        assert settings.CALLBACK_RETRY_DELAY == 1.0


class TestVisualizationSettings:
    """Test VisualizationSettings class."""

    def test_init_defaults(self):
        """Test VisualizationSettings with default values."""
        from src.config.settings import VisualizationSettings

        settings = VisualizationSettings()

        assert settings.VISUALIZE is True
        assert settings.DISPLAY_FPS is True
        assert settings.SHOW_BOX is True
        assert settings.SHOW_TEXT is True
        assert settings.FONT_SCALE == 0.7


class TestLoggingSettings:
    """Test LoggingSettings class."""

    def test_init_defaults(self):
        """Test LoggingSettings with default values."""
        from src.config.settings import LoggingSettings

        settings = LoggingSettings()

        assert settings.LOG_LEVEL == "INFO"
        assert settings.DEBUG is False
        assert settings.USE_COLOR is True


class TestSettings:
    """Test Settings class."""

    def test_init(self):
        """Test Settings initialization."""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.models is not None
        assert settings.camera is not None
        assert settings.runtime is not None
        assert settings.events is not None
        assert settings.visualization is not None
        assert settings.logging is not None

    def test_from_env(self):
        """Test Settings.from_env class method."""
        from src.config.settings import Settings

        settings = Settings.from_env()
        assert settings is not None

    def test_get_gpu_device_cpu(self):
        """Test get_gpu_device returns cpu when configured."""
        from src.config.settings import Settings

        settings = Settings()
        settings.models.YOLO_DEVICE = "cpu"

        assert settings.get_gpu_device() == "cpu"

    def test_get_gpu_device_int(self):
        """Test get_gpu_device returns int for GPU."""
        from src.config.settings import Settings

        settings = Settings()
        settings.models.YOLO_DEVICE = "0"

        assert settings.get_gpu_device() == 0

    def test_get_paddle_device(self):
        """Test get_paddle_device returns configured device."""
        from src.config.settings import Settings

        settings = Settings()
        settings.models.PADDLE_DEVICE = "gpu:1"

        assert settings.get_paddle_device() == "gpu:1"


class TestSettingsFunctions:
    """Test settings module functions."""

    def test_get_settings(self):
        """Test get_settings returns global instance."""
        from src.config import settings as config_module
        from src.config.settings import get_settings

        # Reset global
        config_module._settings = None

        settings = get_settings()
        assert settings is not None

    def test_reload_settings(self):
        """Test reload_settings resets global instance."""
        from src.config import settings as config_module
        from src.config.settings import get_settings, reload_settings

        # Reset global
        config_module._settings = None

        settings1 = get_settings()
        settings2 = reload_settings()

        # Should be different instances
        assert settings1 is not settings2
