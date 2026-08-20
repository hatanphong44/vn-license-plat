"""Models package.

This package contains isolated model adapters following PLAN.md:
- Each model is in its own module
- Models handle load/inference only
- No camera, HTTP, FastAPI, Docker, or DB logic
"""

from .plate_detector import (
    PlateDetectorBase,
    YOLOPlateDetector,
    create_plate_detector,
)
from .text_detector import (
    PaddleTextDetector,
    TextDetectorBase,
    create_text_detector,
)
from .text_recognizer import (
    PaddleTextRecognizer,
    TextRecognizerBase,
    create_text_recognizer,
)

__all__ = [
    "PaddleTextDetector",
    "PaddleTextRecognizer",
    # Plate detector
    "PlateDetectorBase",
    # Text detector
    "TextDetectorBase",
    # Text recognizer
    "TextRecognizerBase",
    "YOLOPlateDetector",
    "create_plate_detector",
    "create_text_detector",
    "create_text_recognizer",
]
