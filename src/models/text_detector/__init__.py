"""Text detector model adapter using PaddleOCR.

Responsibilities (per PLAN.md):
- load/inference only
- No camera, HTTP, FastAPI, Docker, or DB logic
"""

import json
import logging
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
        device: str = "gpu:0",
    ):
        """Initialize PaddleOCR text detector.

        Args:
            device: Device for inference (gpu:0, gpu:1, cpu)
        """
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the model into memory."""
        logger.info("Loading PP-OCRv6_small_det...")
        self._model = TextDetection(
            model_name="PP-OCRv6_small_det",
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
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        det_results = list(
            self._model.predict(
                input=image,
                batch_size=1,
            )
        )

        if not det_results:
            return []

        data = _parse_result(det_results[0])
        polygons = data.get("dt_polys", [])
        scores = data.get("dt_scores", [])

        detections = []
        for poly, score in zip(polygons, scores):
            polygon = np.asarray(poly, dtype=np.float32)
            detections.append(DomainTextDetection.from_polygon_score(polygon, score))

        # Sort by reading order: top-to-bottom, left-to-right
        detections.sort(key=lambda d: (d.y_center, d.x_center))

        return detections


def create_text_detector(
    device: str = "gpu:0",
) -> PaddleTextDetector:
    """Factory function to create text detector.

    Args:
        device: Device for inference

    Returns:
        Configured text detector instance
    """
    detector = PaddleTextDetector(device=device)
    detector.load()
    return detector
