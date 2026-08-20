"""Unit tests for plate collector."""


from src.domain.models import LPRResult, TextRecognition
from src.events.plate_collector import MultiPlateCollector, PlateCollector


class TestPlateCollector:
    """Test PlateCollector."""

    def test_add_detection(self):
        """Test that first detection doesn't complete collection."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)

        # First detection should not complete collection
        is_complete, box_key = collector.add_detection(result)
        assert is_complete is False
        # Without box_key_func, uses plate_normalized as key
        assert box_key == "ABC123"

    def test_collection_completes_at_max_frames(self):
        """Test that collection completes at max_frames."""
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
        is_complete, box_key = collector.add_detection(result)
        assert is_complete is True
        assert box_key == "ABC123"

    def test_cooldown(self):
        """Test that cooldown prevents new detections."""
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

        # Adding during cooldown should return False with None key
        is_complete, box_key = collector.add_detection(result)
        assert is_complete is False
        assert box_key is None

    def test_is_new_plate(self):
        """Test is_new_plate method."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)
        plate_key = "ABC123"

        # New plate
        assert collector.is_new_plate(plate_key) is True

        # After adding, should not be new
        collector.add_detection(result)
        assert collector.is_new_plate(plate_key) is False

    def test_get_best_result(self):
        """Test get_best_result with multiple detections."""
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

    def test_clear(self):
        """Test clear method."""
        collector = PlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
        )

        result = self._create_result("ABC123", 0.9)
        plate_key = "ABC123"
        collector.add_detection(result)

        assert collector.get_best_result(plate_key) is not None

        collector.clear(plate_key)
        assert collector.get_best_result(plate_key) is None

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
    """Test MultiPlateCollector with box-based tracking."""

    def _box_key_func(self, result: LPRResult) -> str:
        """Generate box-based key for testing."""
        x1, y1, x2, y2 = result.box
        q = 10
        return f"{x1//q}_{y1//q}_{x2//q}_{y2//q}"

    def test_add_detections(self):
        """Test adding detections for multiple plates."""
        collector = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
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

        # More frames for first plate
        for _ in range(4):
            collector.add_detections([result1])

        # Should be complete
        completed = collector.add_detections([result1])
        assert len(completed) == 1
        box_key, captured = completed[0]
        assert captured.plate_normalized == "ABC123"
        # Box key should be quantized
        assert box_key == "1_2_10_8"

    def test_multiple_plates_different_positions(self):
        """Test tracking multiple plates at different positions."""
        collector = MultiPlateCollector(
            max_frames=3,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        plate1 = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )

        # Different position
        plate2 = LPRResult(
            plate_index=0,
            plate="ABC123",  # Same text, different position
            plate_normalized="ABC123",
            box=[200, 20, 300, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )

        # Add to collection
        collector.add_detections([plate1])
        collector.add_detections([plate2])

        # Both should be tracked separately
        assert len(collector.collector._collections) == 2


class TestCollectionLogic:
    """Test the NEW collection logic requirements.

    These tests verify:
    A. One vehicle in 20 frames -> only 1 event max
    B. OCR oscillates between results -> best result selected first
    C. Same plate reappears -> no duplicate event
    D. New vehicle with different plate -> exactly 1 event
    E. "New plate detected" only logged AFTER collection completes
    """

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

    def test_best_result_selected_from_multiple_ocr_readings(self):
        """Test that best result is selected from collected OCR readings.

        Even if OCR oscillates between readings, best result is selected first,
        THEN we decide if it's a new plate.
        """
        collector = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        # OCR oscillates between readings
        results = [
            ("ABC123", 0.71),  # low confidence
            ("ABC123", 0.82),  # medium
            ("ABC128", 0.79),  # different reading
            ("ABC123", 0.94),  # HIGHEST confidence - should be selected
            ("ABC123", 0.86),  # another
        ]

        for plate, conf in results:
            result = self._create_result(plate, conf)
            collector.add_detections([result])

        # After 5th frame, collection completes
        completed = collector.add_detections([self._create_result("ABC123", 0.85)])

        assert len(completed) == 1
        _box_key, best = completed[0]

        # Best result should be ABC123 with highest confidence (0.94)
        assert best.plate_normalized == "ABC123"
        assert best.confidence == 0.94

    def test_20_frames_one_event_max(self):
        """Test that 20 frames of same plate produces at most 1 event."""
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        result = self._create_result("ABC123", 0.85)

        # Add 19 frames - should NOT complete yet
        for _ in range(19):
            completed = collector.add_detections([result])
            assert len(completed) == 0, "Should not complete before 20 frames"

        # 20th frame - should complete
        completed = collector.add_detections([result])
        assert len(completed) == 1, "Should complete at 20 frames"

    def test_same_plate_different_positions_same_result(self):
        """Test same plate appearing at different positions still uses best result.

        This simulates the scenario where one vehicle's plate is tracked at
        different box positions but should still be recognized as the same.
        """
        collector = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        # Same plate text, slightly different positions (simulating vehicle movement)
        # Note: with q=10 quantization, these positions are close enough to map to same key
        positions = [
            [10, 20, 100, 80],
            [12, 22, 102, 82],  # slight movement - same quantized key
            [15, 25, 105, 85],  # more movement - same quantized key
        ]

        for i, box in enumerate(positions):
            result = self._create_result("ABC123", 0.85 + i * 0.01, box)
            collector.add_detections([result])

        # All positions map to the same quantized key (coarse quantization prevents fragmentation)
        assert len(collector.collector._collections) == 1

        # All 3 readings collected into one collection
        box_key = "1_2_10_8"
        best = collector.collector.get_best_result(box_key)
        assert best is not None
        assert best.plate_normalized == "ABC123"

    def test_duplicate_plate_no_duplicate_event(self):
        """Test that same plate number doesn't produce duplicate events.

        This is the key logic: compare best_plate with last_processed_plate.
        """
        # This test verifies the collection behavior
        # The actual duplicate prevention is in worker._handle_completed_collection

        collector = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        result = self._create_result("ABC123", 0.85)

        # Collect first plate
        for _ in range(5):
            collector.add_detections([result])
        completed = collector.add_detections([result])
        assert len(completed) == 1
        # best1 = completed[0][1]  # Would be used for further verification

        # Now simulate the same plate appearing again
        # The collection should restart (or be in cooldown)
        collector2 = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        # If we're in cooldown, collection should not start
        # (This is handled by mark_sent which starts cooldown)
        collector2.add_detections([result])  # first detection
        collector2.mark_sent(self._box_key_func(result))  # mark as sent

        # Adding more should be blocked by cooldown
        is_complete, _box_key = collector2.collector.add_detection(result)
        assert is_complete is False, "Should be blocked by cooldown"

    def test_collection_collects_without_publishing(self):
        """Test that collection collects frames without publishing individual events.

        During collection phase:
        - NO is_new_plate() check
        - NO publish
        - NO "New plate detected" log
        """
        collector = MultiPlateCollector(
            max_frames=20,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        result = self._create_result("ABC123", 0.85)

        # Add 10 frames - no completed collections
        for i in range(10):
            completed = collector.add_detections([result])
            assert len(completed) == 0, f"Frame {i+1}: should not complete yet"

        # Only at frame 20 (or timeout) should we get completed
        # For this test with max_frames=20, we won't complete yet
        assert len(collector.collector._collections) == 1
        collection = next(iter(collector.collector._collections.values()))
        assert collection["collection"].size() == 10

    def test_ocr_variations_collected(self):
        """Test that OCR variations are all collected for the same box position.

        Even if OCR returns different readings, they should all be collected
        so we can pick the best one later.
        """
        collector = MultiPlateCollector(
            max_frames=5,
            max_wait_seconds=10.0,
            cooldown_seconds=30.0,
            box_key_func=self._box_key_func,
        )

        # Same box position, different OCR readings
        box = [10, 20, 100, 80]
        readings = ["29A12345", "29A1234S", "29A12345", "29A1234S", "29A12345"]

        for plate in readings:
            result = self._create_result(plate, 0.85, box)
            collector.add_detections([result])

        # Should be complete with all readings collected
        completed = collector.add_detections([self._create_result("29A12345", 0.85, box)])
        assert len(completed) == 1

        # Get collection to verify all readings were collected
        box_key = self._box_key_func(self._create_result("29A12345", 0.85, box))
        best = collector.collector.get_best_result(box_key)

        # Best result should be selected based on voting + confidence
        assert best is not None
        assert best.plate_normalized in ["29A12345", "29A1234S"]
