"""Plate Collector - Collect frames and select best result.

Responsibilities (per PLAN.md):
- Collect up to MAX_CAPTURE_FRAMES (default 20) detections of same plate
- After collection completes, score all results and return the best one
- Handle cooldown after sending
"""

import logging
import time
from collections import Counter
from typing import Optional

from src.domain.models import (
    CapturedPlate,
    PlateCollection,
    LPRResult,
)


logger = logging.getLogger("lpr.events.plate_collector")


class PlateCollector:
    """Collect plate detections and select best result.

    Best-result selection algorithm (per PLAN.md):
    1. Count occurrences of each plate text
    2. Pick the one with most votes
    3. If tie, break by confidence score
    4. If still tie, pick longest text
    """

    def __init__(
        self,
        max_frames: int = 20,
        max_wait_seconds: float = 10.0,
        cooldown_seconds: float = 30.0,
    ):
        """Initialize plate collector.

        Args:
            max_frames: Maximum frames to collect per plate
            max_wait_seconds: Max time to wait before sending (timeout)
            cooldown_seconds: Cooldown after sending
        """
        self.max_frames = max_frames
        self.max_wait_seconds = max_wait_seconds
        self.cooldown_seconds = cooldown_seconds

        self._collections: dict[str, PlateCollection] = {}
        self._cooldowns: dict[str, float] = {}  # plate -> last sent timestamp

    def add_detection(
        self,
        result: LPRResult,
        frame: Optional[object] = None,
    ) -> bool:
        """Add a plate detection to collector.

        Args:
            result: LPR result
            frame: Optional frame image

        Returns:
            True if collection completed (should send event)
        """
        plate_text = result.plate_normalized

        # Check cooldown
        if self.is_in_cooldown(plate_text):
            return False

        # Get or create collection
        collection = self.get_or_create_collection(plate_text)

        # Add capture
        capture = CapturedPlate(
            plate_normalized=plate_text,
            plate=result.plate,
            confidence=result.get_confidence(),
            yolo_score=result.yolo_score,
            box=result.box,
            ocr_results=result.ocr_results,
        )
        collection.add(capture)

        logger.debug(f"Collection progress: plate={plate_text} "
                    f"frames={collection.size()}/{self.max_frames}")

        # Check if complete
        if self.is_complete(plate_text):
            return True

        return False

    def get_or_create_collection(self, plate_text: str) -> PlateCollection:
        """Get existing collection or create new one.

        Args:
            plate_text: Normalized plate text

        Returns:
            Plate collection
        """
        if plate_text not in self._collections:
            # Clear old collections for different plates
            # (keep only the one we're actively collecting)
            self._collections.clear()
            self._collections[plate_text] = PlateCollection(
                plate_number=plate_text,
                max_frames=self.max_frames,
            )

        return self._collections[plate_text]

    def get_collection(self, plate_text: str) -> Optional[PlateCollection]:
        """Get collection for plate.

        Args:
            plate_text: Normalized plate text

        Returns:
            Collection if exists
        """
        return self._collections.get(plate_text)

    def is_complete(self, plate_text: str) -> bool:
        """Check if collection is complete.

        Args:
            plate_text: Normalized plate text

        Returns:
            True if collection should be sent
        """
        collection = self.get_collection(plate_text)
        if not collection:
            return False

        # Check if full
        if collection.is_full():
            return True

        # Check timeout
        if collection.should_timeout(self.max_wait_seconds):
            return True

        return False

    def is_in_cooldown(self, plate_text: str) -> bool:
        """Check if plate is in cooldown.

        Args:
            plate_text: Normalized plate text

        Returns:
            True if in cooldown
        """
        if plate_text not in self._cooldowns:
            return False

        elapsed = time.time() - self._cooldowns[plate_text]
        return elapsed < self.cooldown_seconds

    def get_best_result(self, plate_text: str) -> Optional[CapturedPlate]:
        """Get best result from collection.

        Args:
            plate_text: Normalized plate text

        Returns:
            Best captured plate, or None
        """
        collection = self.get_collection(plate_text)
        if not collection:
            return None

        return collection.get_best_result()

    def get_all_results(self) -> dict[str, list[CapturedPlate]]:
        """Get all results grouped by plate text.

        Returns:
            Dict mapping plate text to list of captures
        """
        result = {}
        for plate_text, collection in self._collections.items():
            if collection.captures:
                result[plate_text] = collection.captures.copy()
        return result

    def should_start_new_collection(self, plate_text: str) -> bool:
        """Check if we should start a new collection.

        Args:
            plate_text: Normalized plate text

        Returns:
            True if should start new collection
        """
        # Already in cooldown
        if self.is_in_cooldown(plate_text):
            return False

        # Already collecting this plate
        if plate_text in self._collections:
            return False

        return True

    def mark_sent(self, plate_text: str) -> None:
        """Mark plate as sent, starting cooldown.

        Args:
            plate_text: Normalized plate text
        """
        self._cooldowns[plate_text] = time.time()

        # Clean up old cooldowns
        self._cleanup_cooldowns()

    def clear(self, plate_text: Optional[str] = None) -> None:
        """Clear collections.

        Args:
            plate_text: Specific plate to clear, or None to clear all
        """
        if plate_text:
            self._collections.pop(plate_text, None)
        else:
            self._collections.clear()

    def _cleanup_cooldowns(self) -> None:
        """Remove stale cooldown entries."""
        now = time.time()
        cutoff = now - self.cooldown_seconds * 10

        stale = [p for p, t in self._cooldowns.items() if t < cutoff]
        for p in stale:
            del self._cooldowns[p]

    def is_new_plate(self, plate_text: str) -> bool:
        """Check if this is a new plate (not in any collection).

        Args:
            plate_text: Normalized plate text

        Returns:
            True if new plate
        """
        # Not in cooldown
        if self.is_in_cooldown(plate_text):
            return False

        # Not in any collection
        if plate_text in self._collections:
            collection = self._collections[plate_text]
            # Collection exists but empty means new plate
            return collection.size() == 0

        return True


class MultiPlateCollector:
    """Collector for tracking multiple plates simultaneously.

    Useful when multiple plates may appear in same frame.
    """

    def __init__(
        self,
        max_frames: int = 20,
        max_wait_seconds: float = 10.0,
        cooldown_seconds: float = 30.0,
    ):
        """Initialize multi-plate collector.

        Args:
            max_frames: Maximum frames per plate
            max_wait_seconds: Max wait before sending
            cooldown_seconds: Cooldown after sending
        """
        self.collector = PlateCollector(
            max_frames=max_frames,
            max_wait_seconds=max_wait_seconds,
            cooldown_seconds=cooldown_seconds,
        )

    def add_detections(
        self,
        results: list[LPRResult],
    ) -> list[str]:
        """Add multiple detections from a frame.

        Args:
            results: List of LPR results from frame

        Returns:
            List of plate texts that completed collection
        """
        completed = []

        for result in results:
            plate_text = result.plate_normalized

            # Check if this is a new plate
            if self.collector.should_start_new_collection(plate_text):
                logger.info(f"New plate detected: {plate_text}")

            # Add to collection
            is_complete = self.collector.add_detection(result)

            if is_complete:
                completed.append(plate_text)

        return completed

    def get_best_result(self, plate_text: str) -> Optional[CapturedPlate]:
        """Get best result for plate."""
        return self.collector.get_best_result(plate_text)

    def mark_sent(self, plate_text: str) -> None:
        """Mark plate as sent."""
        self.collector.mark_sent(plate_text)

    def is_in_cooldown(self, plate_text: str) -> bool:
        """Check if plate is in cooldown."""
        return self.collector.is_in_cooldown(plate_text)

    def clear(self) -> None:
        """Clear all collections."""
        self.collector.clear()
