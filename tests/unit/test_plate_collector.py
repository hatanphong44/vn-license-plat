"""Unit tests for plate collector."""

import pytest
import time

from src.domain.models import LPRResult, TextRecognition
from src.events.plate_collector import PlateCollector, MultiPlateCollector


class TestPlateCollector:
    """Test PlateCollector."""

    def test_add_detection(self):
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)

        # First detection should not complete collection
        is_complete = collector.add_detection(result)
        assert is_complete is False
        assert collector.get_collection("ABC123").size() == 1

    def test_collection_completes_at_max_frames(self):
        collector = PlateCollector(
            max_frames=3,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)

        # Add 2 frames
        collector.add_detection(result)
        collector.add_detection(result)

        # 3rd frame should complete
        is_complete = collector.add_detection(result)
        assert is_complete is True

    def test_cooldown(self):
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)

        # Add and mark as sent
        collector.add_detection(result)
        collector.mark_sent("ABC123")

        # Should be in cooldown
        assert collector.is_in_cooldown("ABC123") is True

        # Adding during cooldown should return False
        is_complete = collector.add_detection(result)
        assert is_complete is False

    def test_should_start_new_collection(self):
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)

        # New plate
        assert collector.should_start_new_collection("ABC123") is True

        # After adding, should not start new
        collector.add_detection(result)
        assert collector.should_start_new_collection("ABC123") is False

    def test_get_best_result(self):
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        # Add some detections for ABC123
        for conf in [0.8, 0.9, 0.85]:
            result = self._create_result("ABC123", conf)
            collector.add_detection(result)

        # Get best result - should be the one with highest confidence (0.9)
        best = collector.get_best_result("ABC123")
        assert best is not None
        assert best.plate_normalized == "ABC123"
        assert best.confidence == 0.9  # Highest confidence for ABC123

    def test_collection_replaced_on_new_plate(self):
        """When a new plate is detected, old collection is replaced."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        # Add detections for ABC123
        result1 = self._create_result("ABC123", 0.9)
        collector.add_detection(result1)

        # Verify ABC123 collection exists
        assert collector.get_collection("ABC123") is not None
        assert collector.get_collection("ABC123").size() == 1

        # Add detection for XYZ789 - this should clear ABC123 collection
        result2 = self._create_result("XYZ789", 0.95)
        collector.add_detection(result2)

        # ABC123 collection should be cleared (only one plate collected at a time)
        assert collector.get_collection("ABC123") is None

        # XYZ789 collection should exist
        assert collector.get_collection("XYZ789") is not None
        assert collector.get_collection("XYZ789").size() == 1

    def test_clear(self):
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)
        collector.add_detection(result)

        assert collector.get_collection("ABC123").size() == 1

        collector.clear("ABC123")
        assert collector.get_collection("ABC123") is None

    def _create_result(self, plate: str, confidence: float) -> LPRResult:
        """Helper to create LPRResult."""
        ocr = TextRecognition(
            text=plate,
            line=0,
            det_score=confidence,
            rec_score=confidence,
            polygon=[],
        )
        return LPRResult(
            plate_index=0,
            plate=plate,
            plate_normalized=plate,
            box=[10, 20, 100, 80],
            yolo_score=confidence,
            class_name="plate",
            ocr_results=[ocr],
        )


class TestMultiPlateCollector:
    """Test MultiPlateCollector."""

    def test_add_detections(self):
        collector = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result1 = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )

        result2 = LPRResult(
            plate_index=0,
            plate="XYZ789",
            plate_normalized="XYZ789",
            box=[200, 20, 300, 80],
            yolo_score=0.85,
            class_name="plate",
            ocr_results=[],
        )

        # First frame with two plates
        completed = collector.add_detections([result1, result2])
        assert len(completed) == 0  # Not complete yet

        # More frames for ABC123
        for _ in range(4):
            collector.add_detections([result1])

        # Should be complete
        completed = collector.add_detections([result1])
        assert "ABC123" in completed
