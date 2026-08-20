"""LPR Runtime - 24/7 License Plate Recognition System.

Note: Core ML libraries (ultralytics, paddleocr) are optional.
Import them explicitly when needed.
"""

__version__ = "2.0.0"

# Domain models - always available
from src.domain.models import (
    CapturedPlate,
    LPRResult,
    PlateCollection,
    PlateDetection,
    PlateEvent,
    TextDetection,
    TextRecognition,
)

# Events
from src.events.plate_collector import PlateCollector

# Pipeline components
from src.pipeline.postprocessor import LPRPostProcessor

__all__ = [
    # Domain
    "CapturedPlate",
    # Pipeline
    "LPRPostProcessor",
    "LPRResult",
    "PlateCollection",
    # Events
    "PlateCollector",
    "PlateDetection",
    "PlateEvent",
    "TextDetection",
    "TextRecognition",
    # Version
    "__version__",
]
