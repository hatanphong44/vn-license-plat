"""Configuration settings for LPR Runtime.

Following PLAN.md: all configuration via environment variables.
No hard-coded values for model paths, camera URLs, GPU, confidence/IoU, etc.
"""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class ModelSettings(BaseSettings):
    """Model configuration."""

    # Base directory for models
    MODEL_BASE_DIR: str = Field(
        default="./models",
        description="Base directory for all models"
    )

    # YOLO plate detector
    YOLO_MODEL_PATH: str = Field(
        default="./models/Plate.pt",
        description="Path to YOLO plate detector model"
    )
    YOLO_DEVICE: str = Field(
        default="0",
        description="GPU device for YOLO (0, 1, cpu)"
    )
    YOLO_CONF: float = Field(
        default=0.25,
        description="YOLO confidence threshold"
    )
    YOLO_IOU: float = Field(
        default=0.45,
        description="YOLO IoU threshold for NMS"
    )

    # PaddleOCR
    PADDLE_DEVICE: str = Field(
        default="gpu:0",
        description="Device for PaddleOCR (gpu:0, gpu:1, cpu)"
    )
    PADDLE_MODEL_DIR: str = Field(
        default="./models",
        description="Directory to save/load PaddleOCR models"
    )
    OCR_UPSCALE: int = Field(
        default=4,
        description="Upscale factor for plate before OCR"
    )
    TEXT_PADDING: int = Field(
        default=10,
        description="Padding around detected text regions"
    )
    REC_MIN_SCORE: float = Field(
        default=0.0,
        description="Minimum recognition score threshold"
    )

    # Model test mode
    TEST_VIDEO: str | None = Field(
        default=None,
        description="Path to test video (None = use camera)"
    )
    TEST_OUTPUT: str | None = Field(
        default=None,
        description="Path to save output video"
    )
    TEST_SHOW_STAGES: bool = Field(
        default=True,
        description="Show intermediate results"
    )
    TEST_SAVE_FRAME: bool = Field(
        default=False,
        description="Save best frames to disk"
    )

    class Config:
        env_prefix = ""


class CameraSettings(BaseSettings):
    """Camera configuration."""

    CAMERA_SOURCE: str = Field(
        default="0",
        description="Camera source (0, 1, or RTSP URL)"
    )
    INFERENCE_FPS: float = Field(
        default=5.0,
        description="Target inference FPS"
    )
    CAMERA_RECONNECT_DELAY: int = Field(
        default=3,
        description="Seconds to wait before reconnecting"
    )
    CAMERA_CONNECT_TIMEOUT: int = Field(
        default=10,
        description="Camera connection timeout in seconds"
    )
    CAMERA_BUFFER_SIZE: int = Field(
        default=1,
        description="Camera buffer size"
    )

    class Config:
        env_prefix = ""


class RuntimeSettings(BaseSettings):
    """Runtime worker configuration."""

    PLATE_COOLDOWN_SECONDS: float = Field(
        default=30.0,
        description="Cooldown after sending plate event"
    )
    MAX_CAPTURE_FRAMES: int = Field(
        default=20,
        description="Maximum frames to collect per plate"
    )
    MAX_CAPTURE_WAIT_SECONDS: float = Field(
        default=10.0,
        description="Max time to wait before sending (timeout)"
    )

    class Config:
        env_prefix = ""


class EventSettings(BaseSettings):
    """Event publisher configuration."""

    CALLBACK_URL: str = Field(
        default="",
        description="HTTP callback URL for plate events"
    )
    CALLBACK_TIMEOUT: float = Field(
        default=5.0,
        description="HTTP callback timeout in seconds"
    )
    CALLBACK_RETRY_COUNT: int = Field(
        default=3,
        description="Number of retry attempts for failed callbacks"
    )
    CALLBACK_RETRY_DELAY: float = Field(
        default=1.0,
        description="Delay between retry attempts in seconds"
    )

    class Config:
        env_prefix = ""


class VisualizationSettings(BaseSettings):
    """Visualization configuration."""

    VISUALIZE: bool = Field(
        default=True,
        description="Enable/disable overlay"
    )
    DISPLAY_FPS: bool = Field(
        default=True,
        description="Show FPS counter"
    )
    SHOW_BOX: bool = Field(
        default=True,
        description="Show detection boxes"
    )
    SHOW_TEXT: bool = Field(
        default=True,
        description="Show plate text"
    )
    FONT_SCALE: float = Field(
        default=0.7,
        description="Text size scale"
    )
    BOX_COLOR: tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="Bounding box color (BGR)"
    )
    TEXT_COLOR: tuple[int, int, int] = Field(
        default=(255, 255, 255),
        description="Text color (BGR)"
    )

    class Config:
        env_prefix = ""


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    USE_COLOR: bool = Field(
        default=True,
        description="Use colored output"
    )

    class Config:
        env_prefix = ""


class Settings:
    """Combined settings container."""

    def __init__(self):
        self.models = ModelSettings()
        self.camera = CameraSettings()
        self.runtime = RuntimeSettings()
        self.events = EventSettings()
        self.visualization = VisualizationSettings()
        self.logging = LoggingSettings()

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls()

    def get_gpu_device(self) -> int | str:
        """Get GPU device for YOLO."""
        device = self.models.YOLO_DEVICE
        if device.lower() == "cpu":
            return "cpu"
        return int(device)

    def get_paddle_device(self) -> str:
        """Get device for PaddleOCR."""
        return self.models.PADDLE_DEVICE


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings.from_env()
    return _settings
