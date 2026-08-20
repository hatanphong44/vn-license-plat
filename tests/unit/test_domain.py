"""Unit tests for domain models."""


from src.domain.models import (
    CapturedPlate,
    LPRResult,
    PlateCollection,
    PlateDetection,
    PlateEvent,
    TextDetection,
    TextRecognition,
)


class TestPlateDetection:
    """Test PlateDetection model."""

    def test_create(self):
        detection = PlateDetection(
            box=[10, 20, 100, 80],
            score=0.95,
            class_id=0,
            class_name="plate",
        )
        assert detection.box == [10, 20, 100, 80]
        assert detection.score == 0.95
        assert detection.class_name == "plate"


class TestTextDetection:
    """Test TextDetection model."""

    def test_from_polygon_score(self):
        import numpy as np
        polygon = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], dtype=np.float32)
        detection = TextDetection.from_polygon_score(polygon, 0.92)

        assert detection.score == 0.92
        assert detection.x_center == 50.0
        assert detection.y_center == 15.0


class TestLPRResult:
    """Test LPRResult model."""

    def test_has_text(self):
        result = LPRResult(
            plate_index=0,
            plate="29A12345",
            plate_normalized="29A12345",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )
        assert result.has_text() is True

    def test_has_no_text(self):
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

    def test_get_confidence(self):
        ocr_result = TextRecognition(
            text="ABC",
            line=0,
            det_score=0.9,
            rec_score=0.85,
            polygon=[],
        )
        result = LPRResult(
            plate_index=0,
            plate="ABC",
            plate_normalized="ABC",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[ocr_result],
        )
        confidence = result.get_confidence()
        # (0.9 + 0.85) / 2 = 0.875
        assert 0.87 < confidence < 0.88


class TestPlateCollection:
    """Test PlateCollection model."""

    def test_add_and_size(self):
        collection = PlateCollection(plate_number="29A12345", max_frames=20)

        capture = CapturedPlate(
            plate_normalized="29A12345",
            plate="29A12345",
            confidence=0.9,
            yolo_score=0.85,
            box=[10, 20, 100, 80],
            ocr_results=[],
        )

        collection.add(capture)
        assert collection.size() == 1
        assert collection.is_full() is False

    def test_max_frames(self):
        collection = PlateCollection(plate_number="ABC", max_frames=3)

        for _i in range(5):
            capture = CapturedPlate(
                plate_normalized="ABC",
                plate="ABC",
                confidence=0.9,
                yolo_score=0.85,
                box=[10, 20, 100, 80],
                ocr_results=[],
            )
            collection.add(capture)

        assert collection.size() == 3
        assert collection.is_full() is True

    def test_get_text_counts(self):
        collection = PlateCollection(plate_number="ABC", max_frames=20)

        # Add 3 captures of "ABC"
        for _ in range(3):
            capture = CapturedPlate(
                plate_normalized="ABC",
                plate="ABC",
                confidence=0.9,
                yolo_score=0.85,
                box=[10, 20, 100, 80],
                ocr_results=[],
            )
            collection.add(capture)

        # Add 2 captures of "ABD"
        for _ in range(2):
            capture = CapturedPlate(
                plate_normalized="ABD",
                plate="ABD",
                confidence=0.9,
                yolo_score=0.85,
                box=[10, 20, 100, 80],
                ocr_results=[],
            )
            collection.add(capture)

        counts = collection.get_text_counts()
        assert counts["ABC"] == 3
        assert counts["ABD"] == 2

    def test_get_best_result(self):
        collection = PlateCollection(plate_number="ABC", max_frames=20)

        # Add captures with different confidences
        for conf in [0.8, 0.9, 0.85, 0.95, 0.7]:
            capture = CapturedPlate(
                plate_normalized="ABC",
                plate="ABC",
                confidence=conf,
                yolo_score=0.85,
                box=[10, 20, 100, 80],
                ocr_results=[],
            )
            collection.add(capture)

        best = collection.get_best_result()
        assert best is not None
        assert best.confidence == 0.95


class TestPlateEvent:
    """Test PlateEvent model."""

    def test_from_result(self):
        ocr_result = TextRecognition(
            text="ABC",
            line=0,
            det_score=0.9,
            rec_score=0.85,
            polygon=[],
        )
        capture = CapturedPlate(
            plate_normalized="ABC",
            plate="ABC",
            confidence=0.85,
            yolo_score=0.9,
            box=[10, 20, 100, 80],
            ocr_results=[ocr_result],
        )

        event = PlateEvent.from_result(capture, "rtsp:stream", 20)

        assert event.camera == "rtsp:stream"
        assert event.plate == "ABC"
        assert event.plate_normalized == "ABC"
        assert event.frames_collected == 20
        assert event.best_confidence == 0.85
        assert event.timestamp != ""

    def test_to_dict(self):
        event = PlateEvent(
            timestamp="2024-01-01T00:00:00",
            camera="0",
            plate="ABC",
            plate_normalized="ABC",
            yolo_score=0.9,
            box=[10, 20, 100, 80],
            frames_collected=20,
            best_confidence=0.85,
        )

        data = event.to_dict()

        assert data["event"] == "new_plate"
        assert data["camera"] == "0"
        assert data["plate"] == "ABC"
        assert data["frames_collected"] == 20
