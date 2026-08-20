"""Plate Collector - Collect frames and select best result.

Responsibilities (per PLAN.md):
- Collect up to MAX_CAPTURE_FRAMES (default 20) detections of same plate
- After collection completes, score all results and return the best one
- Handle cooldown after sending
"""

import logging
import time
from collections.abc import Callable

from src.domain.models import (
    CapturedPlate,
    LPRResult,
    PlateCollection,
)

logger = logging.getLogger("lpr.events.plate_collector")


class PlateCollector:
    """Collect plate detections and select best result.

    Tracks plates by BOUNDING BOX position, not OCR text.
    This allows collecting all OCR variations of the same physical plate.

    Best-result selection algorithm:
    1. Collect all OCR readings of the same plate position
    2. Count occurrences of each plate text
    3. Pick the one with most votes
    4. If tie, break by confidence score
    """

    def __init__(
        self,
        max_frames: int = 20,
        max_wait_seconds: float = 10.0,
        cooldown_seconds: float = 30.0,
        box_key_func: Callable | None = None,
    ):
        """Initialize plate collector.

        Args:
            max_frames: Maximum frames to collect per plate
            max_wait_seconds: Max time to wait before sending (timeout)
            cooldown_seconds: Cooldown after sending
            box_key_func: Function to generate key from LPRResult (for tracking by box)
        """
        self.max_frames = max_frames
        self.max_wait_seconds = max_wait_seconds
        self.cooldown_seconds = cooldown_seconds
        self.box_key_func = box_key_func

        # Track by box_key -> {plate_text, collection, first_detected_time}
        self._collections: dict[str, dict] = {}
        self._cooldowns: dict[str, float] = {}  # box_key -> last sent timestamp

    def add_detection(self, result: LPRResult) -> tuple[bool, str | None]:
        """Add a plate detection to collector.

        Args:
            result: LPR result

        Returns:
            Tuple of (is_complete, box_key)
        """
        plate_text = result.plate_normalized

        # Generate box key
        box_key = self._get_box_key(result)

        # Check cooldown
        if self.is_in_cooldown(box_key):
            return False, None

        # Get or create collection for this box position
        collection_data = self._collections.get(box_key)
        if collection_data is None:
            # New plate at this position
            collection_data = {
                "plate_text": plate_text,  # First seen text
                "collection": PlateCollection(
                    plate_number=plate_text,
                    max_frames=self.max_frames,
                ),
                "first_seen": time.time(),
            }
            self._collections[box_key] = collection_data

        collection = collection_data["collection"]

        # Add capture with frames count
        capture = CapturedPlate(
            plate_normalized=plate_text,
            plate=result.plate,
            confidence=result.get_confidence(),
            yolo_score=result.yolo_score,
            box=result.box,
            ocr_results=result.ocr_results,
            frames_count=collection.size() + 1,  # +1 because add() hasn't been called yet
        )
        collection.add(capture)

        logger.debug(f"Collecting: box={box_key}, frames={collection.size()}/{self.max_frames}")

        # Check if complete
        is_complete = self._is_complete(box_key)
        return is_complete, box_key

    def _get_box_key(self, result: LPRResult) -> str:
        """Generate key for tracking this plate position."""
        if self.box_key_func:
            return self.box_key_func(result)
        # Default: use normalized plate text
        return result.plate_normalized

    def _is_complete(self, box_key: str) -> bool:
        """Check if collection is complete."""
        collection_data = self._collections.get(box_key)
        if not collection_data:
            return False

        collection = collection_data["collection"]

        # Check if full
        if collection.is_full():
            return True

        # Check timeout
        return bool(collection.should_timeout(self.max_wait_seconds))

    def is_in_cooldown(self, box_key: str) -> bool:
        """Check if plate position is in cooldown."""
        if box_key not in self._cooldowns:
            return False
        elapsed = time.time() - self._cooldowns[box_key]
        return elapsed < self.cooldown_seconds

    def get_best_result(self, box_key: str) -> CapturedPlate | None:
        """Get best result from collection."""
        collection_data = self._collections.get(box_key)
        if not collection_data:
            return None
        return collection_data["collection"].get_best_result()

    def mark_sent(self, box_key: str) -> None:
        """Mark plate as sent, starting cooldown."""
        self._cooldowns[box_key] = time.time()
        # Clean up collection
        self._collections.pop(box_key, None)
        self._cleanup_cooldowns()

    def clear(self, box_key: str | None = None) -> None:
        """Clear collections."""
        if box_key:
            self._collections.pop(box_key, None)
        else:
            self._collections.clear()

    def _cleanup_cooldowns(self) -> None:
        """Remove stale cooldown entries."""
        now = time.time()
        cutoff = now - self.cooldown_seconds * 10
        stale = [k for k, t in self._cooldowns.items() if t < cutoff]
        for k in stale:
            del self._cooldowns[k]

    def is_new_plate(self, box_key: str) -> bool:
        """Check if this is a new plate position."""
        return box_key not in self._collections and not self.is_in_cooldown(box_key)


class MultiPlateCollector:
    """Collector for tracking multiple plates simultaneously.

    Tracks plates by BOUNDING BOX position, not OCR text.
    This allows collecting all OCR variations of the same physical plate.
    """

    def __init__(
        self,
        max_frames: int = 20,
        max_wait_seconds: float = 10.0,
        cooldown_seconds: float = 30.0,
        box_key_func: Callable | None = None,
    ):
        """Initialize multi-plate collector.

        Args:
            max_frames: Maximum frames per plate
            max_wait_seconds: Max wait before sending
            cooldown_seconds: Cooldown after sending
            box_key_func: Function to generate key from LPRResult (for tracking by box)
        """
        self.collector = PlateCollector(
            max_frames=max_frames,
            max_wait_seconds=max_wait_seconds,
            cooldown_seconds=cooldown_seconds,
            box_key_func=box_key_func,
        )

    def add_detections(
        self,
        results: list[LPRResult],
    ) -> list[tuple[str, CapturedPlate]]:
        """Add multiple detections from a frame.

        Args:
            results: List of LPR results from frame

        Returns:
            List of (box_key, best_result) tuples for completed collections
        """
        completed = []

        for result in results:
            is_complete, box_key = self.collector.add_detection(result)
            if is_complete:
                best = self.collector.get_best_result(box_key)
                if best:
                    completed.append((box_key, best))

        return completed

    def mark_sent(self, box_key: str) -> None:
        """Mark plate as sent."""
        self.collector.mark_sent(box_key)

    def clear(self) -> None:
        """Clear all collections."""
        self.collector.clear()
