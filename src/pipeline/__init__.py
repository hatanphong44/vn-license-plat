"""Pipeline package."""

from .cropper import (
    PlateCropper,
    PlatePreprocessor,
    TextCropper,
)
from .postprocessor import (
    LPRPostProcessor,
    MultiLineConcatenator,
    TextNormalizer,
)

# Note: LPRPipeline requires ML models (ultralytics, paddleocr)
# Import explicitly: from src.pipeline.lpr_pipeline import LPRPipeline

__all__ = [
    "LPRPostProcessor",
    "MultiLineConcatenator",
    # Cropping
    "PlateCropper",
    "PlatePreprocessor",
    "TextCropper",
    # Postprocessing
    "TextNormalizer",
]
