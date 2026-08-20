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
    TextDetectorBase,
    PaddleTextDetector,
    create_text_detector,
)
from .text_recognizer import (
    TextRecognizerBase,
    PaddleTextRecognizer,
    create_text_recognizer,
)

__all__ = [
    # Plate detector
    "PlateDetectorBase",
    "YOLOPlateDetector",
    "create_plate_detector",
    # Text detector
    "TextDetectorBase",
    "PaddleTextDetector",
    "create_text_detector",
    # Text recognizer
    "TextRecognizerBase",
    "PaddleTextRecognizer",
    "create_text_recognizer",
]
