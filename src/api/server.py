"""API Server - FastAPI endpoints.

Responsibilities (per PLAN.md):
- /health, /ready, /metrics, optional /predict
- No camera loop
"""

import logging
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

        return {
            "status": "ready" if runtime_ready else "starting",
            "runtime_running": runtime_ready,
            "camera_source": None,  # Would need camera ref
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

    @app.post("/stop")
    def stop():
        """Stop the runtime."""
        controller = app.state.controller
        controller.stop()
        return {"status": "stopped"}

    @app.post("/start")
    def start():
        """Check runtime status."""
        controller = app.state.controller
        return {
            "running": controller.is_running,
        }


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
