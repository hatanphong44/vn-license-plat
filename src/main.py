"""Main entry point for LPR Runtime.

Usage:
    python -m src.main
    python -m src.main --debug
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import create_app
from src.camera import create_camera
from src.config import Settings, get_settings
from src.events import create_http_publisher
from src.logging import get_logger, setup_logging
from src.models import (
    create_plate_detector,
    create_text_detector,
    create_text_recognizer,
)
from src.pipeline.lpr_pipeline import create_pipeline
from src.runtime import LPRRuntimeWorker, WorkerConfig


def load_models(settings: Settings) -> tuple:
    """Load all ML models.

    Args:
        settings: Application settings

    Returns:
        Tuple of (plate_detector, text_detector, text_recognizer)
    """
    logger = get_logger("main")

    logger.info(f"Loading plate detector: {settings.models.YOLO_MODEL_PATH}")
    plate_detector = create_plate_detector(
        model_path=settings.models.YOLO_MODEL_PATH,
        conf=settings.models.YOLO_CONF,
        iou=settings.models.YOLO_IOU,
        device=settings.models.YOLO_DEVICE,
    )

    logger.info("Loading text detector (PP-OCRv6_small_det)...")
    text_detector = create_text_detector(
        model_dir=settings.models.PADDLE_MODEL_DIR,
        device=settings.models.PADDLE_DEVICE,
    )

    logger.info("Loading text recognizer (PP-OCRv6_small_rec)...")
    text_recognizer = create_text_recognizer(
        model_dir=settings.models.PADDLE_MODEL_DIR,
        device=settings.models.PADDLE_DEVICE,
    )

    logger.info("All models loaded successfully")

    return plate_detector, text_detector, text_recognizer


def _log_debug_startup(logger, settings: Settings, debug: bool) -> None:
    """Log debug startup information.

    Args:
        logger: Logger instance
        settings: Application settings
        debug: Whether debug mode is enabled
    """
    if not debug:
        return

    logger.info("")
    logger.info("=" * 56)
    logger.info("  LPR DEBUG MODE")
    logger.info("=" * 56)
    logger.info("")

    # Camera info
    logger.info("  Camera")
    logger.info(f"    source: {settings.camera.CAMERA_SOURCE}")
    logger.info(f"    buffer size: {settings.camera.CAMERA_BUFFER_SIZE}")
    logger.info("")

    # Models info
    logger.info("  Models")
    logger.info(f"    Plate detector: {settings.models.YOLO_MODEL_PATH}")
    logger.info(f"    Text detector: {settings.models.PADDLE_MODEL_DIR}/PP-OCRv6_small_det")
    logger.info(f"    Text recognizer: {settings.models.PADDLE_MODEL_DIR}/PP-OCRv6_small_rec")
    logger.info("")

    # Device info
    logger.info("  Devices")
    logger.info(f"    YOLO: {settings.models.YOLO_DEVICE}")
    logger.info(f"    OCR: {settings.models.PADDLE_DEVICE}")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info(f"    CUDA available: {cuda_available}")
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"    CUDA device {i}: {props.name}")
    except ImportError:
        pass

    try:
        import cv2
        logger.info(f"    OpenCV: {cv2.__version__}")
    except ImportError:
        pass

    logger.info("")

    # Runtime config
    logger.info("  Runtime")
    logger.info(f"    result window: 3.0s")
    logger.info(f"    consensus ratio: 0.3")
    logger.info("")

    logger.info("=" * 56)
    logger.info("")


def create_runtime(
    settings: Settings,
    plate_detector,
    text_detector,
    text_recognizer,
    preview: bool = False,
) -> LPRRuntimeWorker:
    """Create and configure the runtime worker."""
    get_logger("main")

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

    camera = create_camera(
        source=settings.camera.CAMERA_SOURCE,
        buffer_size=settings.camera.CAMERA_BUFFER_SIZE,
        timeout=settings.camera.CAMERA_CONNECT_TIMEOUT,
        reconnect_delay=settings.camera.CAMERA_RECONNECT_DELAY,
    )

    publisher = create_http_publisher(
        url=settings.events.CALLBACK_URL,
        timeout=settings.events.CALLBACK_TIMEOUT,
        retry_count=settings.events.CALLBACK_RETRY_COUNT,
        retry_delay=settings.events.CALLBACK_RETRY_DELAY,
    )

    worker_config = WorkerConfig(
        reconnect_delay=settings.camera.CAMERA_RECONNECT_DELAY,
        preview=preview,
    )

    overlay = None
    if preview:
        from src.visualization import create_overlay_renderer
        overlay = create_overlay_renderer(
            enabled=True,
            display_fps=True,
            window_name="LPR Camera Preview (q to quit)",
            headless=True,
            save_dir="captures",
            save_interval_seconds=5,
        )

    worker = LPRRuntimeWorker(
        camera=camera,
        pipeline=pipeline,
        publisher=publisher,
        config=worker_config,
        overlay=overlay,
    )

    if overlay is not None:
        overlay.start()

    return worker


# Lazy app initialization for uvicorn
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create FastAPI application."""
    global _app
    if _app is None:
        from src.api import create_app

        settings = get_settings()
        setup_logging(level="INFO")

        plate_detector, text_detector, text_recognizer = load_models(settings)

        pipeline = create_pipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        _app = create_app(pipeline=pipeline)

    return _app


# Module-level app for uvicorn
app = get_app()


def run_runtime(preview: bool = False, debug: bool = False):
    """Run the LPR runtime.

    Args:
        preview: Enable camera preview window
        debug: Enable debug profiling
    """
    settings = get_settings()
    debug_mode = debug or settings.logging.DEBUG

    setup_logging(
        level=settings.logging.LOG_LEVEL,
        debug=debug_mode,
        use_color=settings.logging.USE_COLOR,
    )

    # Initialize profiler
    from src.observability import init_profiler, set_profiler_enabled
    init_profiler(enabled=debug_mode)
    set_profiler_enabled(debug_mode)

    logger = get_logger("main")

    # Print startup info
    logger.info("=" * 56)
    logger.info("LPR Runtime starting")
    logger.info(f"Camera: {settings.camera.CAMERA_SOURCE}")
    logger.info(f"Result window: 3.0s (continuous inference)")
    if debug_mode:
        logger.info("DEBUG: enabled")
    logger.info("=" * 56)

    # Log debug startup info
    _log_debug_startup(logger, settings, debug_mode)

    try:
        plate_detector, text_detector, text_recognizer = load_models(settings)

        worker = create_runtime(
            settings=settings,
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
            preview=preview,
        )

        worker.start()

        logger.info("Runtime started. Press Ctrl+C to stop.")
        if preview:
            logger.info("Preview: enabled")
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

    settings = get_settings()
    setup_logging(level="INFO")
    logger = get_logger("api")

    plate_detector, text_detector, text_recognizer = load_models(settings)

    pipeline = create_pipeline(
        plate_detector=plate_detector,
        text_detector=text_detector,
        text_recognizer=text_recognizer,
    )

    app = create_app(pipeline=pipeline)

    logger.info("Starting API server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LPR Runtime")
    parser.add_argument("--mode", choices=["runtime", "api"], default="runtime",
                       help="Run mode: runtime (default) or api")
    parser.add_argument("--preview", action="store_true",
                       help="Enable camera preview window")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug profiling with periodic performance summaries")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    else:
        run_runtime(preview=args.preview, debug=args.debug)
