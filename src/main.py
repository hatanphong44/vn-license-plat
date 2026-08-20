"""Main entry point for LPR Runtime.

Usage:
    python -m src.main
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import logging
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings, Settings
from src.logging import setup_logging, get_logger
from src.models import (
    create_plate_detector,
    create_text_detector,
    create_text_recognizer,
)
from src.camera import create_camera
from src.pipeline.lpr_pipeline import LPRPipeline, create_pipeline
from src.events import create_http_publisher
from src.runtime import LPRRuntimeWorker, WorkerConfig, get_controller
from src.api import create_app
from src.visualization import create_overlay_renderer


def load_models(settings: Settings) -> tuple:
    """Load all ML models.

    Args:
        settings: Application settings

    Returns:
        Tuple of (plate_detector, text_detector, text_recognizer)
    """
    logger = get_logger("main")

    # Load plate detector
    logger.info(f"Loading plate detector: {settings.models.YOLO_MODEL_PATH}")
    plate_detector = create_plate_detector(
        model_path=settings.models.YOLO_MODEL_PATH,
        conf=settings.models.YOLO_CONF,
        iou=settings.models.YOLO_IOU,
        device=settings.models.YOLO_DEVICE,
    )

    # Load text detector
    logger.info("Loading text detector (PP-OCRv6_small_det)...")
    text_detector = create_text_detector(
        device=settings.models.PADDLE_DEVICE,
    )

    # Load text recognizer
    logger.info("Loading text recognizer (PP-OCRv6_small_rec)...")
    text_recognizer = create_text_recognizer(
        device=settings.models.PADDLE_DEVICE,
    )

    logger.info("All models loaded successfully")

    return plate_detector, text_detector, text_recognizer


def create_runtime(
    settings: Settings,
    plate_detector,
    text_detector,
    text_recognizer,
) -> LPRRuntimeWorker:
    """Create and configure the runtime worker.

    Args:
        settings: Application settings
        plate_detector: Plate detection model
        text_detector: Text detection model
        text_recognizer: Text recognition model

    Returns:
        Configured runtime worker
    """
    logger = get_logger("main")

    # Create pipeline
    pipeline = create_pipeline(
        plate_detector=plate_detector,
        text_detector=text_detector,
        text_recognizer=text_recognizer,
        config={
            "plate_padding": 0.05,
            "text_padding": settings.models.TEXT_PADDING,
            "upscale_factor": settings.models.OCR_UPSCALE,
            "rec_min_score": settings.models.REC_MIN_SCORE,
        },
    )

    # Create camera
    camera = create_camera(
        source=settings.camera.CAMERA_SOURCE,
        buffer_size=settings.camera.CAMERA_BUFFER_SIZE,
        timeout=settings.camera.CAMERA_CONNECT_TIMEOUT,
        reconnect_delay=settings.camera.CAMERA_RECONNECT_DELAY,
    )

    # Create event publisher
    publisher = create_http_publisher(
        url=settings.events.CALLBACK_URL,
        timeout=settings.events.CALLBACK_TIMEOUT,
        retry_count=settings.events.CALLBACK_RETRY_COUNT,
        retry_delay=settings.events.CALLBACK_RETRY_DELAY,
    )

    # Create worker config
    worker_config = WorkerConfig(
        inference_fps=settings.camera.INFERENCE_FPS,
        reconnect_delay=settings.camera.CAMERA_RECONNECT_DELAY,
        max_frames=settings.runtime.MAX_CAPTURE_FRAMES,
        max_wait_seconds=settings.runtime.MAX_CAPTURE_WAIT_SECONDS,
        cooldown_seconds=settings.runtime.PLATE_COOLDOWN_SECONDS,
    )

    # Create runtime worker
    worker = LPRRuntimeWorker(
        camera=camera,
        pipeline=pipeline,
        publisher=publisher,
        config=worker_config,
    )

    return worker


def run_runtime():
    """Run the LPR runtime."""
    # Setup logging
    settings = get_settings()
    setup_logging(
        level=settings.logging.LOG_LEVEL,
        debug=settings.logging.DEBUG,
        use_color=settings.logging.USE_COLOR,
    )
    logger = get_logger("main")

    # Print startup info
    logger.info("=" * 70)
    logger.info("LPR Runtime starting")
    logger.info(f"Camera: {settings.camera.CAMERA_SOURCE}")
    logger.info(f"Inference FPS: {settings.camera.INFERENCE_FPS}")
    logger.info(f"Callback URL: {settings.events.CALLBACK_URL or '(not configured)'}")
    logger.info("=" * 70)

    try:
        # Load models
        plate_detector, text_detector, text_recognizer = load_models(settings)

        # Create runtime
        worker = create_runtime(
            settings=settings,
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        # Start runtime
        worker.start()

        # Keep main thread alive
        logger.info("Runtime started. Press Ctrl+C to stop.")
        while True:
            import time
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if 'worker' in locals():
            worker.stop()
        logger.info("Shutdown complete")


def run_api():
    """Run the API server."""
    import uvicorn

    # Setup
    settings = get_settings()
    setup_logging(level="INFO")
    logger = get_logger("api")

    # Load models
    plate_detector, text_detector, text_recognizer = load_models(settings)

    # Create pipeline
    pipeline = create_pipeline(
        plate_detector=plate_detector,
        text_detector=text_detector,
        text_recognizer=text_recognizer,
    )

    # Create app
    app = create_app(pipeline=pipeline)

    logger.info("Starting API server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LPR Runtime")
    parser.add_argument("--mode", choices=["runtime", "api"], default="runtime",
                       help="Run mode: runtime (default) or api")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    else:
        run_runtime()
