"""Pipeline package."""

from .cropper import (
    PlateCropper,
    TextCropper,
    PlatePreprocessor,
)
from .postprocessor import (
    TextNormalizer,
    MultiLineConcatenator,
    LPRPostProcessor,
)
# Note: LPRPipeline requires ML models (ultralytics, paddleocr)
# Import explicitly: from src.pipeline.lpr_pipeline import LPRPipeline

__all__ = [
    # Cropping
    "PlateCropper",
    "TextCropper",
    "PlatePreprocessor",
    # Postprocessing
    "TextNormalizer",
    "MultiLineConcatenator",
    "LPRPostProcessor",
]
