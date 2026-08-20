"""Text recognizer model adapter using PaddleOCR.

Responsibilities (per PLAN.md):
- load/inference only
- No camera, HTTP, FastAPI, Docker, or DB logic
"""

import os
import json
import logging
from typing import Protocol
import numpy as np
from paddleocr import TextRecognition

from src.domain.models import TextRecognition as DomainTextRecognition


logger = logging.getLogger("lpr.models.text_recognizer")


class TextRecognizerBase(Protocol):
    """Protocol for text recognizer models."""

    def recognize(self, images: list[np.ndarray]) -> list[DomainTextRecognition]:
        """Recognize text in cropped images."""
        ...


def _parse_result(res) -> dict:
    """Parse PaddleOCR result to extract recognition data."""
    data = res.json
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data


class PaddleTextRecognizer:
    """PaddleOCR-based text recognizer."""

    def __init__(
        self,
        model_dir: str = "./models",
        device: str = "gpu:0",
    ):
        """Initialize PaddleOCR text recognizer.

        Args:
            model_dir: Base directory containing PP-OCRv6_small_rec folder
            device: Device for inference (gpu:0, gpu:1, cpu)
        """
        self.model_dir = model_dir
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the model into memory."""
        # model_dir should be the base models directory containing PP-OCRv6_small_rec folder
        rec_model_dir = os.path.join(self.model_dir, "PP-OCRv6_small_rec")
        logger.info(f"Loading PP-OCRv6_small_rec from {rec_model_dir}...")
        self._model = TextRecognition(
            model_name="PP-OCRv6_small_rec",
            model_dir=rec_model_dir,
            device=self.device,
        )
        logger.info("OCR text recognizer loaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def recognize(
        self,
        images: list[np.ndarray],
        min_score: float = 0.0,
    ) -> list[DomainTextRecognition]:
        """Recognize text in cropped images.

        Args:
            images: List of cropped text region images
            min_score: Minimum recognition score threshold

        Returns:
            List of TextRecognition results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not images:
            return []

        rec_results = list(
            self._model.predict(
                input=images,
                batch_size=len(images),
            )
        )

        outputs = []
        for res in rec_results:
            data = _parse_result(res)

            text = str(data.get("rec_text", "")).strip()
            try:
                rec_score = float(data.get("rec_score", 0.0))
            except Exception:
                rec_score = 0.0

            if text and rec_score >= min_score:
                outputs.append(DomainTextRecognition(
                    text=text,
                    line=len(outputs),  # Reading order
                    det_score=0.0,  # Set by text detector
                    rec_score=rec_score,
                    polygon=[],  # Set by text detector
                ))

        return outputs


def create_text_recognizer(
    model_dir: str = "./models",
    device: str = "gpu:0",
) -> PaddleTextRecognizer:
    """Factory function to create text recognizer.

    Args:
        model_dir: Base directory containing model folders
        device: Device for inference

    Returns:
        Configured text recognizer instance
    """
    recognizer = PaddleTextRecognizer(model_dir=model_dir, device=device)
    recognizer.load()
    return recognizer
