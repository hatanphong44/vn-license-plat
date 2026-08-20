"""LPR postprocessing utilities.

Responsibilities (per PLAN.md):
- Text concatenation for multi-line plates
- Normalization and validation
- Result formatting
"""

import logging
import re

from src.domain.models import LPRResult, TextRecognition

logger = logging.getLogger("lpr.pipeline.postprocessor")


class TextNormalizer:
    """Normalize and validate plate text."""

    # Valid plate characters (Vietnamese format)
    VALID_CHARS = re.compile(r"^[A-Z0-9]+$")

    def __init__(self, min_length: int = 4, max_length: int = 15):
        """Initialize text normalizer.

        Args:
            min_length: Minimum valid plate text length
            max_length: Maximum valid plate text length
        """
        self.min_length = min_length
        self.max_length = max_length

    def normalize(self, text: str) -> str:
        """Normalize plate text.

        - Converts to uppercase
        - Removes spaces, dots, hyphens, etc.
        - Keeps only letters, numbers, and underscores (for concatenation)

        Args:
            text: Raw OCR text

        Returns:
            Normalized text
        """
        text = str(text).upper().strip()
        return "".join(ch for ch in text if ch.isalnum() or ch == '_')

    def is_valid(self, text: str) -> bool:
        """Check if plate text is valid.

        Args:
            text: Normalized plate text (may contain underscore for multi-line)

        Returns:
            True if valid
        """
        if not text:
            return False

        # Remove underscores for validation (they're for concatenation only)
        text_no_underscore = text.replace("_", "")

        if len(text_no_underscore) < self.min_length or len(text_no_underscore) > self.max_length:
            return False

        return bool(self.VALID_CHARS.match(text_no_underscore))

    def clean_results(self, results: list[TextRecognition]) -> list[TextRecognition]:
        """Filter and clean OCR results.

        Args:
            results: Raw OCR results

        Returns:
            Cleaned results with valid text only
        """
        cleaned = []
        for result in results:
            normalized = self.normalize(result.text)
            # Only filter if text is empty or has invalid characters
            # Don't filter based on length - length validation happens after concatenation
            if normalized and self.VALID_CHARS.match(normalized):
                # Update with normalized text
                result.text = normalized
                cleaned.append(result)

        return cleaned


class MultiLineConcatenator:
    """Concatenate multi-line plate text.

    Following PLAN.md:
    - If 2 lines → concatenate with "_" (e.g., "29A" + "12345" → "29A_12345")
    - If 1 line → keep as-is
    - Order: top → bottom
    """

    def concatenate(self, results: list[TextRecognition]) -> str:
        """Concatenate multiple text lines.

        Args:
            results: Sorted OCR results (by reading order)

        Returns:
            Concatenated plate text
        """
        if not results:
            return ""

        # Sort by line number (already sorted, but ensure consistency)
        sorted_results = sorted(results, key=lambda r: r.line)

        if len(sorted_results) == 1:
            return sorted_results[0].text

        # Concatenate with underscore
        parts = [r.text for r in sorted_results]
        return "_".join(parts)

    def concatenate_raw(self, texts: list[str]) -> str:
        """Concatenate raw text strings.

        Args:
            texts: List of text strings (in reading order)

        Returns:
            Concatenated text
        """
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]
        return "_".join(texts)


class LPRPostProcessor:
    """Main postprocessor for LPR results."""

    def __init__(
        self,
        min_length: int = 4,
        max_length: int = 15,
    ):
        """Initialize LPR postprocessor.

        Args:
            min_length: Minimum valid plate length
            max_length: Maximum valid plate length
        """
        self.normalizer = TextNormalizer(min_length, max_length)
        self.concatenator = MultiLineConcatenator()

    def process(self, ocr_results: list[TextRecognition]) -> str | None:
        """Process OCR results to get final plate text.

        Args:
            ocr_results: Raw OCR results from text recognizer

        Returns:
            Processed plate text, or None if invalid
        """
        if not ocr_results:
            return None

        # Clean invalid results
        cleaned = self.normalizer.clean_results(ocr_results)

        if not cleaned:
            return None

        # Concatenate multi-line text
        plate_text = self.concatenator.concatenate(cleaned)

        # Final validation
        normalized = self.normalizer.normalize(plate_text)
        if not self.normalizer.is_valid(normalized):
            return None

        return normalized

    def process_result(self, result: LPRResult) -> str | None:
        """Process LPRResult to get final plate text.

        Args:
            result: Full LPR result

        Returns:
            Processed plate text, or None if invalid
        """
        return self.process(result.ocr_results)

    def format_for_output(
        self,
        plate_text: str,
        normalized: bool = True,
    ) -> dict:
        """Format plate text for output.

        Args:
            plate_text: Plate text
            normalized: Whether text is normalized

        Returns:
            Formatted output dict
        """
        return {
            "plate": plate_text,
            "plate_normalized": plate_text.upper() if normalized else self.normalizer.normalize(plate_text),
            "valid": self.normalizer.is_valid(plate_text),
        }
