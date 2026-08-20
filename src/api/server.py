"""API Server - FastAPI endpoints.

Responsibilities (per PLAN.md):
- /health, /ready, /metrics, optional /predict
- Camera runtime control via /start-runtime, /stop-runtime
"""

import logging
import threading
import time
from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline.lpr_pipeline import LPRPipeline
from src.runtime.controller import RuntimeController, get_controller

logger = logging.getLogger("lpr.api")


def create_app(
    controller: RuntimeController | None = None,
    pipeline: LPRPipeline | None = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        controller: Runtime controller
        pipeline: LPR pipeline for /predict endpoint

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="LPR Runtime API",
        description="24/7 License Plate Recognition Runtime",
        version="2.0.0",
    )

    # Store references
    app.state.controller = controller or get_controller()
    app.state.pipeline = pipeline
    app.state.start_time = time.time()
    app.state.camera = None
    app.state.worker = None
    app.state.runtime_thread = None

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/ready")
    def ready():
        """Readiness check endpoint."""
        controller = app.state.controller
        runtime_ready = controller.is_running
        camera = getattr(app.state, "camera", None)
        camera_source = camera.source if camera else None

        return {
            "status": "ready" if runtime_ready else "starting",
            "runtime_running": runtime_ready,
            "camera_source": camera_source,
            "models": {
                "plate_detection": "loaded",
                "ocr_detection": "loaded",
                "ocr_recognition": "loaded",
            },
        }

    @app.get("/metrics")
    def metrics():
        """Metrics endpoint."""
        controller = app.state.controller
        uptime = time.time() - app.state.start_time

        return {
            "uptime_seconds": uptime,
            "runtime_stats": controller.get_stats(),
        }

    @app.get("/")
    def root():
        """Root endpoint."""
        return {
            "name": "LPR Runtime API",
            "version": "2.0.0",
            "docs": "/docs",
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):  # noqa: B008
        """Manual single-image test endpoint."""
        pipeline = app.state.pipeline

        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Pipeline not configured",
            )

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail="Only image uploads are supported",
            )

        # Read and decode image
        raw = await file.read()
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot decode image",
            )

        # Run inference
        results = pipeline.process_frame(image)

        return {
            "plates": [
                {
                    "plate": r.plate,
                    "plate_normalized": r.plate_normalized,
                    "box": r.box,
                    "yolo_score": r.yolo_score,
                    "confidence": r.get_confidence(),
                    "ocr_results": [
                        {
                            "text": ocr.text,
                            "line": ocr.line,
                            "score": ocr.rec_score,
                        }
                        for ocr in r.ocr_results
                    ],
                }
                for r in results
            ],
            "count": len(results),
        }

    @app.post("/start-runtime")
    def start_runtime():
        """Start the camera runtime for continuous plate detection.

        This endpoint starts the camera and begins the continuous
        plate detection loop in a background thread.
        """
        from src.camera import create_camera
        from src.config import get_settings
        from src.events import create_http_publisher
        from src.runtime import LPRRuntimeWorker, WorkerConfig

        controller = app.state.controller

        if controller.is_running:
            return {
                "status": "already_running",
                "message": "Runtime is already running",
                "camera_source": app.state.camera.source if app.state.camera else None,
            }

        try:
            settings = get_settings()

            # Create camera
            camera = create_camera(
                source=settings.camera.CAMERA_SOURCE,
                buffer_size=settings.camera.CAMERA_BUFFER_SIZE,
                timeout=settings.camera.CAMERA_CONNECT_TIMEOUT,
                reconnect_delay=settings.camera.CAMERA_RECONNECT_DELAY,
            )

            # Test camera connection
            if not camera.connect():
                camera.disconnect()
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to camera: {settings.camera.CAMERA_SOURCE}",
                )

            # Store camera reference
            app.state.camera = camera

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

            # Create worker
            pipeline = app.state.pipeline
            worker = LPRRuntimeWorker(
                camera=camera,
                pipeline=pipeline,
                publisher=publisher,
                config=worker_config,
            )

            app.state.worker = worker

            # Start worker in background thread
            def runtime_loop():
                try:
                    worker.start()
                except Exception as e:
                    logger.error(f"Runtime error: {e}")

            runtime_thread = threading.Thread(target=runtime_loop, daemon=True)
            runtime_thread.start()
            app.state.runtime_thread = runtime_thread

            logger.info("Camera runtime started")
            return {
                "status": "started",
                "message": "Camera runtime started successfully",
                "camera_source": camera.source,
                "inference_fps": settings.camera.INFERENCE_FPS,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to start runtime: {e!s}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start runtime: {e!s}",
            ) from e

    @app.post("/stop-runtime")
    def stop_runtime():
        """Stop the camera runtime."""
        controller = app.state.controller

        if not controller.is_running:
            return {
                "status": "not_running",
                "message": "Runtime is not running",
            }

        try:
            controller.stop()
            if app.state.camera:
                app.state.camera.disconnect()
            app.state.camera = None
            app.state.worker = None
            app.state.runtime_thread = None

            logger.info("Camera runtime stopped")
            return {
                "status": "stopped",
                "message": "Camera runtime stopped",
            }
        except Exception as e:
            logger.error(f"Failed to stop runtime: {e!s}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop runtime: {e!s}",
            ) from e

    @app.post("/start")
    def start():
        """Alias for /start-runtime for backward compatibility."""
        return start_runtime()

    @app.post("/stop")
    def stop():
        """Alias for /stop-runtime for backward compatibility."""
        return stop_runtime()


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
