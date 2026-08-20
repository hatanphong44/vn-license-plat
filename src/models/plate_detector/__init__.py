"""Plate detector model adapter.

Responsibilities (per PLAN.md):
- load/inference only
- No camera, HTTP, FastAPI, Docker, or DB logic
"""

import logging
import time
from typing import Protocol

import numpy as np
import torch
from ultralytics import YOLO

from src.domain.models import PlateDetection

logger = logging.getLogger("lpr.models.plate_detector")


class PlateDetectorBase(Protocol):
    """Protocol for plate detector models."""

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Detect plates in image."""
        ...


class YOLOPlateDetector:
    """YOLO-based plate detector."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "0",
    ):
        """Initialize YOLO plate detector."""
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the model into memory."""
        logger.info(f"Loading YOLO plate detector: {self.model_path}")
        self._model = YOLO(self.model_path)
        logger.info("YOLO plate detector loaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Detect plates in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of PlateDetection results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Synchronize CUDA if using GPU (for accurate timing in debug mode)
        from src.observability import get_profiler
        profiler = get_profiler()

        if profiler.enabled and torch.cuda.is_available() and 'cuda' in str(self.device).lower():
            torch.cuda.synchronize()

        inf_start = time.perf_counter()

        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        if profiler.enabled and torch.cuda.is_available() and 'cuda' in str(self.device).lower():
            torch.cuda.synchronize()

        inf_ms = (time.perf_counter() - inf_start) * 1000

        # Record timing via profiler
        if profiler.enabled:
            profiler.yolo_call(inf_ms)

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()

        detections = []
        for box, score, cls in zip(boxes, scores, classes, strict=False):
            x1, y1, x2, y2 = map(int, box)
            detections.append(PlateDetection(
                box=[x1, y1, x2, y2],
                score=float(score),
                class_id=int(cls),
                class_name=self._model.names[int(cls)],
            ))

        return detections


def create_plate_detector(
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "0",
) -> YOLOPlateDetector:
    """Factory function to create plate detector."""
    detector = YOLOPlateDetector(
        model_path=model_path,
        conf=conf,
        iou=iou,
        device=device,
    )
    detector.load()
    return detector
