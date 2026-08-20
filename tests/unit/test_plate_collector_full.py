"""Unit tests for plate collector - additional edge cases."""

import time

from src.domain.models import LPRResult, TextRecognition
from src.events.plate_collector import MultiPlateCollector, PlateCollector


class TestPlateCollectorEdgeCases:
    """Test PlateCollector edge cases."""

    def _create_result(self, plate: str, confidence: float, box: list[int] | None = None) -> LPRResult:
        """Helper to create LPRResult."""
        ocr = TextRecognition(
            text=plate,
            line=0,
            det_score=confidence,
            rec_score=confidence,
            polygon=[],
        )
        if box is None:
            box = [10, 20, 100, 80]
        return LPRResult(
            plate_index=0,
            plate=plate,
            plate_normalized=plate,
            box=box,
            yolo_score=confidence,
            class_name="plate",
            ocr_results=[ocr],
        )

    def test_is_complete_with_nonexistent_key(self):
        """Test _is_complete returns False for nonexistent key."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )
        result = collector._is_complete("nonexistent_key")
        assert result is False

    def test_clear_specific_key(self):
        """Test clearing specific collection."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )
        result = self._create_result("ABC123", 0.9)
        collector.add_detection(result)
        assert collector.get_best_result("ABC123") is not None

        collector.clear("ABC123")
        assert collector.get_best_result("ABC123") is None

    def test_clear_none_key(self):
        """Test clearing with None clears all collections."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )
        result1 = self._create_result("ABC123", 0.9, [10, 20, 100, 80])
        result2 = self._create_result("XYZ789", 0.9, [200, 20, 300, 80])

        # Need box_key_func to track separately
        collector.box_key_func = lambda r: str(r.box[0])
        collector.add_detection(result1)
        collector.add_detection(result2)

        assert collector.get_best_result("10") is not None
        assert collector.get_best_result("200") is not None

        collector.clear(None)
        assert collector.get_best_result("10") is None
        assert collector.get_best_result("200") is None

    def test_cleanup_cooldowns(self):
        """Test _cleanup_cooldowns removes stale entries."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )
        # Add some cooldowns
        collector._cooldowns["old_key"] = time.time() - 400  # Very old
        collector._cooldowns["new_key"] = time.time() - 10  # Recent

        collector._cleanup_cooldowns()

        # old_key should be removed (400 > 30*10=300)
        # new_key should remain
        assert "old_key" not in collector._cooldowns

    def test_get_best_result_nonexistent_key(self):
        """Test get_best_result returns None for nonexistent key."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )
        result = collector.get_best_result("nonexistent")
        assert result is None

    def test_box_key_func_custom(self):
        """Test custom box_key_func."""
        def custom_key(result: LPRResult) -> str:
            return f"{result.plate_normalized}_custom"

        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=custom_key,
        )

        result = self._create_result("ABC123", 0.9)
        _is_complete, box_key = collector.add_detection(result)
        assert box_key == "ABC123_custom"

    def test_add_detection_same_box_new_plate(self):
        """Test adding detection at same box position with different text."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        # Same box position
        box = [10, 20, 100, 80]
        result1 = self._create_result("ABC123", 0.9, box)
        result2 = self._create_result("XYZ789", 0.9, box)

        collector.add_detection(result1)
        # Without box_key_func, this creates a NEW collection
        _is_complete, box_key = collector.add_detection(result2)
        assert box_key == "XYZ789"

    def test_multiple_plates_same_text_different_box(self):
        """Test tracking multiple plates with same text but different positions."""
        def box_key_func(result: LPRResult) -> str:
            return str(result.box[0] // 50)  # Quantize by x position

        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=box_key_func,
        )

        # Two positions far apart
        result1 = self._create_result("ABC123", 0.9, [10, 20, 100, 80])
        result2 = self._create_result("ABC123", 0.9, [500, 20, 600, 80])

        collector.add_detections([result1])
        collector.add_detections([result2])

        # Should track two separate collections
        assert len(collector.collector._collections) == 2


class TestMultiPlateCollectorEdgeCases:
    """Test MultiPlateCollector edge cases."""

    def _box_key_func(self, result: LPRResult) -> str:
        """Generate box-based key for testing."""
        x1, y1, x2, y2 = result.box
        q = 10
        return f"{x1//q}_{y1//q}_{x2//q}_{y2//q}"

    def _create_result(self, plate: str, confidence: float, box: list[int] | None = None) -> LPRResult:
        """Helper to create LPRResult."""
        ocr = TextRecognition(
            text=plate,
            line=0,
            det_score=confidence,
            rec_score=confidence,
            polygon=[],
        )
        if box is None:
            box = [10, 20, 100, 80]
        return LPRResult(
            plate_index=0,
            plate=plate,
            plate_normalized=plate,
            box=box,
            yolo_score=confidence,
            class_name="plate",
            ocr_results=[ocr],
        )

    def test_add_detections_empty_list(self):
        """Test adding empty list of detections."""
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )
        completed = collector.add_detections([])
        assert completed == []

    def test_add_detections_one_complete(self):
        """Test that add_detections returns completed plates."""
        collector = MultiPlateCollector(
            max_frames=3,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )
        box = [10, 20, 100, 80]

        # Add frames until complete
        for _ in range(3):
            collector.add_detections([self._create_result("ABC123", 0.9, box)])

        # Next frame should return completed
        completed = collector.add_detections([self._create_result("ABC123", 0.9, box)])
        assert len(completed) == 1
        assert completed[0][1].plate_normalized == "ABC123"

    def test_mark_sent(self):
        """Test mark_sent."""
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )
        result = self._create_result("ABC123", 0.9, [10, 20, 100, 80])
        collector.add_detections([result])

        box_key = self._box_key_func(result)
        collector.mark_sent(box_key)

        # Should be in cooldown
        assert collector.collector.is_in_cooldown(box_key)

    def test_clear_all(self):
        """Test clear clears all collections."""
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        result1 = self._create_result("ABC123", 0.9, [10, 20, 100, 80])
        result2 = self._create_result("XYZ789", 0.9, [500, 20, 600, 80])

        collector.add_detections([result1])
        collector.add_detections([result2])

        assert len(collector.collector._collections) == 2

        collector.clear()
        assert len(collector.collector._collections) == 0

    def test_cooldown_prevents_new_collection(self):
        """Test that cooldown prevents starting new collection."""
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )
        result = self._create_result("ABC123", 0.9, [10, 20, 100, 80])
        box_key = self._box_key_func(result)

        # Mark as sent (starts cooldown)
        collector.mark_sent(box_key)

        # Should be in cooldown
        assert collector.collector.is_in_cooldown(box_key) is True

        # Adding should be blocked
        is_complete, returned_key = collector.collector.add_detection(result)
        assert is_complete is False
        assert returned_key is None

    def test_is_new_plate_after_cooldown(self):
        """Test is_new_plate after cooldown expires."""
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=0.001,  # Very short cooldown
            box_key_func=self._box_key_func,
        )
        result = self._create_result("ABC123", 0.9, [10, 20, 100, 80])
        box_key = self._box_key_func(result)

        # Mark as sent
        collector.mark_sent(box_key)

        # Immediately after, should not be new
        assert collector.collector.is_new_plate(box_key) is False

        # Wait for cooldown to expire
        time.sleep(0.01)

        # Now should be new again
        assert collector.collector.is_new_plate(box_key) is True
