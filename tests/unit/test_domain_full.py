"""Unit tests for domain models - additional coverage."""

from src.domain.models import (
    CapturedPlate,
    LPRResult,
    PlateCollection,
    PlateDetection,
    PlateEvent,
    TextDetection,
    TextRecognition,
)


class TestPlateDetectionEdgeCases:
    """Test PlateDetection edge cases."""

    def test_create_with_defaults(self):
        """Test creating PlateDetection with required args."""
        detection = PlateDetection(
            box=[0, 0, 100, 100],
            score=0.9,
            class_id=0,
            class_name="plate",
        )
        assert detection.box == [0, 0, 100, 100]
        assert detection.score == 0.9
        assert detection.class_name == "plate"

    def test_plate_detection_repr(self):
        """Test PlateDetection repr."""
        detection = PlateDetection(
            box=[10, 20, 100, 80],
            score=0.95,
            class_id=0,
            class_name="plate",
        )
        r = repr(detection)
        assert "PlateDetection" in r


class TestTextDetectionEdgeCases:
    """Test TextDetection edge cases."""

    def test_from_polygon_with_array(self):
        """Test creating TextDetection from polygon array."""
        import numpy as np
        polygon = np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float64)
        detection = TextDetection.from_polygon_score(
            polygon=polygon,
            score=0.9,
        )
        assert detection.score == 0.9
        assert len(detection.polygon) == 4
        assert detection.x_center == 20.0
        assert detection.y_center == 30.0


class TestLPRResultEdgeCases:
    """Test LPRResult edge cases."""

    def test_create_minimal(self):
        """Test creating LPRResult with minimal args."""
        result = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )
        assert result.plate == "ABC123"

    def test_has_text_with_empty_plate(self):
        """Test has_text when plate is empty."""
        result = LPRResult(
            plate_index=0,
            plate="",
            plate_normalized="",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )
        assert result.has_text() is False

    def test_get_confidence_no_ocr(self):
        """Test get_confidence when no OCR results."""
        result = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )
        conf = result.get_confidence()
        assert conf == 0.9  # falls back to yolo_score

    def test_get_confidence_with_ocr(self):
        """Test get_confidence with OCR results."""
        ocr = TextRecognition(
            text="ABC",
            line=0,
            det_score=0.9,
            rec_score=0.8,
            polygon=[],
        )
        result = LPRResult(
            plate_index=0,
            plate="ABC",
            plate_normalized="ABC",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[ocr],
        )
        conf = result.get_confidence()
        # (yolo + avg_rec) / 2 = (0.9 + 0.8) / 2 = 0.85
        assert abs(conf - 0.85) < 0.01

    def test_get_confidence_multiple_ocr(self):
        """Test get_confidence with multiple OCR results."""
        ocr1 = TextRecognition(text="A", line=0, det_score=0.9, rec_score=0.7, polygon=[])
        ocr2 = TextRecognition(text="B", line=1, det_score=0.9, rec_score=0.9, polygon=[])
        result = LPRResult(
            plate_index=0,
            plate="AB",
            plate_normalized="AB",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[ocr1, ocr2],
        )
        conf = result.get_confidence()
        # (yolo + avg_rec) / 2 = (0.9 + (0.7+0.9)/2) / 2 = (0.9 + 0.8) / 2 = 0.85
        assert abs(conf - 0.85) < 0.01


class TestPlateCollectionEdgeCases:
    """Test PlateCollection edge cases."""

    def test_add_and_size_empty(self):
        """Test size of empty collection."""
        collection = PlateCollection("ABC123", max_frames=10)
        assert collection.size() == 0

    def test_add_multiple(self):
        """Test adding multiple captures."""
        collection = PlateCollection("ABC123", max_frames=10)
        for i in range(5):
            cap = CapturedPlate(
                plate_normalized="ABC123",
                plate="ABC123",
                confidence=0.9,
                yolo_score=0.9,
                box=[10, 20, 100, 80],
                ocr_results=[],
                frames_count=i + 1,
            )
            collection.add(cap)
        assert collection.size() == 5

    def test_get_text_counts_single_candidate(self):
        """Test get_text_counts with single candidate."""
        collection = PlateCollection("ABC123", max_frames=10)
        for _ in range(3):
            cap = CapturedPlate(
                plate_normalized="ABC123",
                plate="ABC123",
                confidence=0.9,
                yolo_score=0.9,
                box=[10, 20, 100, 80],
                ocr_results=[],
                frames_count=1,
            )
            collection.add(cap)

        counts = collection.get_text_counts()
        assert counts == {"ABC123": 3}

    def test_get_text_counts_multiple_candidates(self):
        """Test get_text_counts with multiple candidates."""
        collection = PlateCollection("ABC123", max_frames=10)
        caps = [
            ("ABC123", 0.9),
            ("ABC123", 0.85),
            ("ABC124", 0.95),  # different reading
        ]
        for i, (plate, conf) in enumerate(caps):
            cap = CapturedPlate(
                plate_normalized=plate,
                plate=plate,
                confidence=conf,
                yolo_score=conf,
                box=[10, 20, 100, 80],
                ocr_results=[],
                frames_count=i + 1,
            )
            collection.add(cap)

        counts = collection.get_text_counts()
        assert counts["ABC123"] == 2
        assert counts["ABC124"] == 1

    def test_get_best_result_single(self):
        """Test get_best_result with single candidate."""
        collection = PlateCollection("ABC123", max_frames=10)
        cap = CapturedPlate(
            plate_normalized="ABC123",
            plate="ABC123",
            confidence=0.9,
            yolo_score=0.9,
            box=[10, 20, 100, 80],
            ocr_results=[],
            frames_count=1,
        )
        collection.add(cap)

        best = collection.get_best_result()
        assert best is not None
        assert best.plate_normalized == "ABC123"

    def test_get_best_result_multiple(self):
        """Test get_best_result selects by frequency then confidence."""
        collection = PlateCollection("ABC123", max_frames=10)
        # Add 3 of ABC123 with low confidence
        for _ in range(3):
            cap = CapturedPlate(
                plate_normalized="ABC123",
                plate="ABC123",
                confidence=0.7,
                yolo_score=0.7,
                box=[10, 20, 100, 80],
                ocr_results=[],
                frames_count=1,
            )
            collection.add(cap)

        # Add 2 of XYZ789 with high confidence
        for _ in range(2):
            cap = CapturedPlate(
                plate_normalized="XYZ789",
                plate="XYZ789",
                confidence=0.95,
                yolo_score=0.95,
                box=[10, 20, 100, 80],
                ocr_results=[],
                frames_count=1,
            )
            collection.add(cap)

        best = collection.get_best_result()
        # ABC123 has more votes (3 vs 2)
        assert best.plate_normalized == "ABC123"
        assert best.confidence == 0.7


class TestPlateEventEdgeCases:
    """Test PlateEvent edge cases."""

    def test_from_result_defaults(self):
        """Test creating PlateEvent with defaults."""
        cap = CapturedPlate(
            plate_normalized="ABC123",
            plate="ABC123",
            confidence=0.9,
            yolo_score=0.9,
            box=[10, 20, 100, 80],
            ocr_results=[],
            frames_count=10,
        )
        event = PlateEvent.from_result(cap, "cam0", 10)
        assert event.plate_normalized == "ABC123"
        assert event.camera == "cam0"
        assert event.frames_collected == 10

    def test_to_dict(self):
        """Test converting PlateEvent to dict."""
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )
        d = event.to_dict()
        assert d["plate"] == "ABC123"
        assert d["plate_normalized"] == "ABC123"
        assert d["camera"] == "cam0"
        assert d["frames_collected"] == 10
        assert "timestamp" in d

    def test_event_repr(self):
        """Test PlateEvent repr."""
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )
        r = repr(event)
        assert "ABC123" in r
