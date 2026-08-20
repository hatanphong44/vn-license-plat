"""Text detector model adapter using PaddleOCR.

Responsibilities (per PLAN.md):
- load/inference only
- No camera, HTTP, FastAPI, Docker, or DB logic
"""

import json
import logging
import os
import time
from typing import Protocol

import numpy as np
from paddleocr import TextDetection

from src.domain.models import TextDetection as DomainTextDetection

logger = logging.getLogger("lpr.models.text_detector")


class TextDetectorBase(Protocol):
    """Protocol for text detector models."""

    def detect(self, image: np.ndarray) -> list[DomainTextDetection]:
        """Detect text regions in image."""
        ...


def _parse_result(res) -> dict:
    """Parse PaddleOCR result to extract detection data."""
    data = res.json
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data


class PaddleTextDetector:
    """PaddleOCR-based text detector."""

    def __init__(
        self,
        model_dir: str = "./models",
        device: str = "gpu:0",
    ):
        """Initialize PaddleOCR text detector."""
        self.model_dir = model_dir
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the model into memory."""
        det_model_dir = os.path.join(self.model_dir, "PP-OCRv6_small_det")
        logger.info(f"Loading PP-OCRv6_small_det from {det_model_dir}...")
        self._model = TextDetection(
            model_name="PP-OCRv6_small_det",
            model_dir=det_model_dir,
            device=self.device,
        )
        logger.info("OCR text detector loaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def detect(self, image: np.ndarray) -> list[DomainTextDetection]:
        """Detect text regions in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of TextDetection results sorted by reading order
        """
        from src.observability import get_profiler

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        profiler = get_profiler()

        inf_start = time.perf_counter()

        det_results = list(
            self._model.predict(
                input=image,
                batch_size=1,
            )
        )

        inf_ms = (time.perf_counter() - inf_start) * 1000

        if profiler.enabled:
            profiler.ocr_detection(inf_ms)

        if not det_results:
            return []

        data = _parse_result(det_results[0])
        polygons = data.get("dt_polys", [])
        scores = data.get("dt_scores", [])

        detections = []
        for poly, score in zip(polygons, scores, strict=False):
            polygon = np.asarray(poly, dtype=np.float32)
            detections.append(DomainTextDetection.from_polygon_score(polygon, score))

        # Sort by reading order: top-to-bottom, left-to-right
        detections.sort(key=lambda d: (d.y_center, d.x_center))

        return detections


def create_text_detector(
    model_dir: str = "./models",
    device: str = "gpu:0",
) -> PaddleTextDetector:
    """Factory function to create text detector."""
    detector = PaddleTextDetector(model_dir=model_dir, device=device)
    detector.load()
    return detector
