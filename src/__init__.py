"""LPR Runtime - 24/7 License Plate Recognition System.

Note: Core ML libraries (ultralytics, paddleocr) are optional.
Import them explicitly when needed.
"""

__version__ = "2.0.0"

# Domain models - always available
from src.domain.models import (
    PlateDetection,
    TextDetection,
    TextRecognition,
    LPRResult,
    CapturedPlate,
    PlateCollection,
    PlateEvent,
)

# Pipeline components
from src.pipeline.postprocessor import LPRPostProcessor

# Events
from src.events.plate_collector import PlateCollector

__all__ = [
    # Version
    "__version__",
    # Domain
    "PlateDetection",
    "TextDetection",
    "TextRecognition",
    "LPRResult",
    "CapturedPlate",
    "PlateCollection",
    "PlateEvent",
    # Pipeline
    "LPRPostProcessor",
    # Events
    "PlateCollector",
]
