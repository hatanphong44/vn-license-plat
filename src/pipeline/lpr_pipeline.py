"""LPR Pipeline - Main inference pipeline.

Responsibilities (per PLAN.md):
- Frame → Detection → Crop → OCR → Postprocess → LPRResult
- No HTTP, no events
"""

import logging
import time

import numpy as np

from src.domain.models import (
    LPRResult,
    PlateDetection,
    TextDetection,
    TextRecognition,
)
from src.models import (
    PlateDetectorBase,
    TextDetectorBase,
    TextRecognizerBase,
)
from src.observability import get_profiler
from src.pipeline.cropper import PlateCropper, PlatePreprocessor, TextCropper
from src.pipeline.postprocessor import LPRPostProcessor

logger = logging.getLogger("lpr.pipeline")


class LPRPipeline:
    """Main LPR inference pipeline.

    Pipeline flow:
    1. Plate Detection (YOLO)
    2. Plate Cropping
    3. Text Detection (PaddleOCR)
    4. Text Cropping
    5. Text Recognition (PaddleOCR)
    6. Postprocessing (concatenation, normalization)
    """

    def __init__(
        self,
        plate_detector: PlateDetectorBase,
        text_detector: TextDetectorBase,
        text_recognizer: TextRecognizerBase,
        plate_padding: float = 0.05,
        text_padding: int = 10,
        upscale_factor: int = 4,
        rec_min_score: float = 0.0,
    ):
        """Initialize LPR pipeline."""
        self.plate_detector = plate_detector
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer

        self.plate_cropper = PlateCropper(padding=plate_padding)
        self.text_cropper = TextCropper(padding=text_padding)
        self.preprocessor = PlatePreprocessor(upscale_factor=upscale_factor)
        self.postprocessor = LPRPostProcessor()

        self.rec_min_score = rec_min_score

    def process_frame(
        self,
        frame: np.ndarray,
        include_frame: bool = False,
    ) -> list[LPRResult]:
        """Process a single frame through the LPR pipeline.

        Args:
            frame: Input frame (BGR format)
            include_frame: Whether to include frame in results

        Returns:
            List of LPR results for detected plates
        """
        profiler = get_profiler()
        results = []

        # Step 1: Detect plates (YOLO)
        plate_start = time.perf_counter()
        plate_detections = self.plate_detector.detect(frame)
        yolo_ms = (time.perf_counter() - plate_start) * 1000

        profiler.yolo_call(yolo_ms)

        # Per-frame logging disabled - only summary logging in debug mode

        # Step 2-6: Process each plate
        for plate_idx, plate_det in enumerate(plate_detections):
            result = self._process_plate(
                frame=frame,
                detection=plate_det,
                plate_idx=plate_idx,
                include_frame=include_frame,
            )

            if result and result.has_text():
                results.append(result)
                # Per-detection logging disabled - only summary logging in debug mode

        return results

    def _process_plate(
        self,
        frame: np.ndarray,
        detection: PlateDetection,
        plate_idx: int,
        include_frame: bool = False,
    ) -> LPRResult | None:
        """Process a single plate detection."""
        # Step 2: Crop plate
        plate_crop = self.plate_cropper.crop(frame, detection)
        if plate_crop is None:
            return None

        # Step 3: Preprocess for OCR
        plate_prep = self.preprocessor.preprocess(plate_crop)

        # Step 4: Detect text regions
        det_start = time.perf_counter()
        text_detections = self.text_detector.detect(plate_prep)
        det_ms = (time.perf_counter() - det_start) * 1000

        get_profiler().ocr_detection(det_ms)

        if not text_detections:
            return None

        # Step 5: Crop and recognize text
        rec_start = time.perf_counter()
        ocr_results = self._recognize_text(plate_prep, text_detections)
        rec_ms = (time.perf_counter() - rec_start) * 1000

        get_profiler().ocr_recognition(rec_ms)

        if not ocr_results:
            return None

        # Log OCR result count
        get_profiler().ocr_result()

        # Step 6: Postprocess
        plate_text = self.postprocessor.process(ocr_results)

        if not plate_text:
            return None

        raw_text = self.postprocessor.concatenator.concatenate(ocr_results)

        return LPRResult(
            plate_index=plate_idx,
            plate=raw_text,
            plate_normalized=plate_text,
            box=detection.box,
            yolo_score=detection.score,
            class_name=detection.class_name,
            ocr_results=ocr_results,
            frame=frame if include_frame else None,
        )

    def _recognize_text(
        self,
        plate: np.ndarray,
        text_detections: list[TextDetection],
    ) -> list[TextRecognition]:
        """Recognize text from plate."""
        # Crop text regions
        crops = []
        metadata = []

        for i, detection in enumerate(text_detections):
            crop = self.text_cropper.crop(plate, detection)
            if crop is None:
                continue

            crops.append(crop)
            metadata.append({
                "line": i,
                "det_score": detection.score,
                "polygon": detection.polygon.tolist(),
            })

        if not crops:
            return []

        # Batch recognize
        recognitions = self.text_recognizer.recognize(
            crops,
            min_score=self.rec_min_score,
        )

        # Attach metadata
        results = []
        for meta, rec in zip(metadata, recognitions, strict=False):
            rec.line = meta["line"]
            rec.det_score = meta["det_score"]
            rec.polygon = meta["polygon"]
            results.append(rec)

        # Sort by reading order
        results.sort(key=lambda r: (r.line, r.x_center if hasattr(r, 'x_center') else 0))

        return results

    def process_single_plate(
        self,
        plate_crop: np.ndarray,
    ) -> str | None:
        """Process a single cropped plate image."""
        plate_prep = self.preprocessor.preprocess(plate_crop)

        text_detections = self.text_detector.detect(plate_prep)

        if not text_detections:
            return None

        ocr_results = self._recognize_text(plate_prep, text_detections)

        if not ocr_results:
            return None

        return self.postprocessor.process(ocr_results)


def create_pipeline(
    plate_detector: PlateDetectorBase,
    text_detector: TextDetectorBase,
    text_recognizer: TextRecognizerBase,
    config: dict | None = None,
) -> LPRPipeline:
    """Factory function to create LPR pipeline."""
    cfg = config or {}

    return LPRPipeline(
        plate_detector=plate_detector,
        text_detector=text_detector,
        text_recognizer=text_recognizer,
        plate_padding=cfg.get("plate_padding", 0.05),
        text_padding=cfg.get("text_padding", 10),
        upscale_factor=cfg.get("upscale_factor", 4),
        rec_min_score=cfg.get("rec_min_score", 0.0),
    )
