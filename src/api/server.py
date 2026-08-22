"""API Server - FastAPI endpoints.

Responsibilities (per PLAN.md):
- /health, /ready, /metrics, optional /predict
- Camera runtime control via /start-runtime, /stop-runtime
"""

import logging
import threading
import time
from datetime import UTC, datetime
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline.lpr_pipeline import LPRPipeline
from src.runtime.controller import RuntimeController, get_controller

logger = logging.getLogger("lpr.api")


class ReadinessState(str, Enum):
    """Readiness state enumeration."""
    INITIALIZING = "initializing"
    MODELS_LOADING = "models_loading"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


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

    # Readiness tracking
    app.state.readiness_state = ReadinessState.INITIALIZING
    app.state.readiness_details = {}

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
        """Health check endpoint - lightweight liveness probe.

        Returns 200 if the process is alive and the API is responding.
        Does NOT verify models or GPU availability.
        """
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/ready")
    def ready():
        """Readiness check endpoint - verifies runtime readiness.

        Checks:
        - Model initialization state
        - GPU availability (when GPU mode is configured)
        - Runtime worker state

        Returns appropriate HTTP status:
        - 200: Service is ready
        - 503: Service is not ready (with details)
        """
        controller = app.state.controller
        pipeline = app.state.pipeline

        # Determine readiness state
        is_ready = True
        status = "ready"
        details = {
            "models": {},
            "gpu": {},
            "runtime": {},
        }

        # Check pipeline and models
        if pipeline is None:
            details["models"]["status"] = "not_configured"
            is_ready = False
            status = "starting"
        else:
            # Check each model component
            plate_detector_ready = pipeline.plate_detector is not None
            text_detector_ready = pipeline.text_detector is not None
            text_recognizer_ready = pipeline.text_recognizer is not None

            details["models"] = {
                "plate_detector": "ready" if plate_detector_ready else "not_loaded",
                "text_detector": "ready" if text_detector_ready else "not_loaded",
                "text_recognizer": "ready" if text_recognizer_ready else "not_loaded",
                "status": "ready" if all([plate_detector_ready, text_detector_ready, text_recognizer_ready]) else "partial",
            }

            if not all([plate_detector_ready, text_detector_ready, text_recognizer_ready]):
                is_ready = False
                status = "models_not_ready"

        # Check GPU availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            details["gpu"]["cuda_available"] = cuda_available
            if cuda_available:
                details["gpu"]["cuda_version"] = torch.version.cuda
                details["gpu"]["gpu_count"] = torch.cuda.device_count()
                if torch.cuda.device_count() > 0:
                    details["gpu"]["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            details["gpu"]["cuda_available"] = False
            details["gpu"]["error"] = "PyTorch not available"

        # Check PaddlePaddle GPU
        try:
            import paddle
            paddle_gpu = paddle.device.is_compiled_with_cuda()
            details["gpu"]["paddle_gpu"] = paddle_gpu
            if paddle_gpu:
                paddle_gpu_count = paddle.device.cuda.device_count()
                details["gpu"]["paddle_gpu_count"] = paddle_gpu_count
        except ImportError:
            details["gpu"]["paddle_gpu"] = False
            details["gpu"]["error"] = "PaddlePaddle not available"

        # Check runtime state
        runtime_ready = controller.is_running
        camera = getattr(app.state, "camera", None)
        camera_source = camera.source if camera else None
        camera_connected = camera.is_connected() if camera else False

        details["runtime"] = {
            "worker_running": runtime_ready,
            "camera_connected": camera_connected,
            "camera_source": camera_source,
        }

        # Determine overall readiness
        runtime_ready_state = runtime_ready or camera_connected

        if status == "ready" and not runtime_ready_state:
            status = "ready_no_camera"
        elif status == "ready" and runtime_ready_state:
            status = "ready"

        # Return appropriate HTTP status
        if not is_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": status,
                    "ready": False,
                    **details,
                },
            )

        return {
            "status": status,
            "ready": True,
            "timestamp": datetime.now(UTC).isoformat(),
            **details,
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
                reconnect_delay=settings.camera.CAMERA_RECONNECT_DELAY,
                preview=False,
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
