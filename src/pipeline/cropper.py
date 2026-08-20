"""Plate and text cropping utilities.

Responsibilities (per PLAN.md):
- Crop plate regions from frames
- Crop text regions for OCR
- Handle padding and preprocessing
"""

import logging

import cv2
import numpy as np

from src.domain.models import PlateDetection, TextDetection

logger = logging.getLogger("lpr.pipeline.cropper")


class PlateCropper:
    """Utility for cropping plate regions from frames."""

    def __init__(self, padding: float = 0.05):
        """Initialize plate cropper.

        Args:
            padding: Padding ratio around plate box (default 5%)
        """
        self.padding = padding

    def crop(
        self,
        frame: np.ndarray,
        detection: PlateDetection,
    ) -> np.ndarray | None:
        """Crop plate region from frame.

        Args:
            frame: Full frame image
            detection: Plate detection result

        Returns:
            Cropped plate image, or None if crop failed
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.box

        # Add padding
        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None

    def crop_with_coords(
        self,
        frame: np.ndarray,
        box: list[int],
    ) -> tuple[np.ndarray, list[int]] | None:
        """Crop region and return with crop coordinates.

        Args:
            frame: Full frame image
            box: Box coordinates [x1, y1, x2, y2]

        Returns:
            Tuple of (cropped image, crop coordinates) or None
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box

        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if not crop.size:
            return None

        return crop, [crop_x1, crop_y1, crop_x2, crop_y2]


class TextCropper:
    """Utility for cropping text regions from plates."""

    def __init__(self, padding: int = 10, border_size: int = 10):
        """Initialize text cropper.

        Args:
            padding: Padding around text region
            border_size: White border size for better OCR
        """
        self.padding = padding
        self.border_size = border_size

    def crop(
        self,
        plate: np.ndarray,
        detection: TextDetection,
    ) -> np.ndarray | None:
        """Crop text region from plate.

        Args:
            plate: Plate image
            detection: Text detection result

        Returns:
            Cropped text region with white border, or None
        """
        polygon = detection.polygon

        x1 = int(np.floor(np.min(polygon[:, 0]))) - self.padding
        y1 = int(np.floor(np.min(polygon[:, 1]))) - self.padding
        x2 = int(np.ceil(np.max(polygon[:, 0]))) + self.padding
        y2 = int(np.ceil(np.max(polygon[:, 1]))) + self.padding

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(plate.shape[1], x2)
        y2 = min(plate.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = plate[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Add white border for better OCR
        return cv2.copyMakeBorder(
            crop,
            self.border_size,
            self.border_size,
            self.border_size,
            self.border_size,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def crop_polygon(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
    ) -> np.ndarray | None:
        """Crop arbitrary polygon region from image.

        Args:
            image: Source image
            polygon: Polygon coordinates [[x1,y1], ...]

        Returns:
            Cropped region with white border, or None
        """
        x1 = int(np.floor(np.min(polygon[:, 0]))) - self.padding
        y1 = int(np.floor(np.min(polygon[:, 1]))) - self.padding
        x2 = int(np.ceil(np.max(polygon[:, 0]))) + self.padding
        y2 = int(np.ceil(np.max(polygon[:, 1]))) + self.padding

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.copyMakeBorder(
            crop,
            self.border_size,
            self.border_size,
            self.border_size,
            self.border_size,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )


class PlatePreprocessor:
    """Preprocessing utilities for plate images before OCR."""

    def __init__(self, upscale_factor: int = 4):
        """Initialize plate preprocessor.

        Args:
            upscale_factor: Upscale factor for OCR
        """
        self.upscale_factor = upscale_factor

    def preprocess(self, plate: np.ndarray) -> np.ndarray:
        """Preprocess plate image for OCR.

        Args:
            plate: Plate image

        Returns:
            Preprocessed plate image
        """
        # Upscale for better OCR accuracy
        if self.upscale_factor > 1:
            plate = cv2.resize(
                plate,
                None,
                fx=self.upscale_factor,
                fy=self.upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        return plate

    def denoise(self, plate: np.ndarray) -> np.ndarray:
        """Apply denoising to plate image.

        Args:
            plate: Plate image

        Returns:
            Denoised plate image
        """
        return cv2.fastNlMeansDenoisingColored(
            plate,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def adjust_contrast(self, plate: np.ndarray) -> np.ndarray:
        """Adjust contrast of plate image.

        Args:
            plate: Plate image

        Returns:
            Contrast-adjusted plate image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        # Merge channels
        lab = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
