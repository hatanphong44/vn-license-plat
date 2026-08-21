"""Tests for src/models/* modules"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPlateDetectorModels:
    """Test plate detector models."""

    def test_yolo_detector_init(self):
        """Test YOLOPlateDetector initialization."""
        from src.models.plate_detector import YOLOPlateDetector

        detector = YOLOPlateDetector(
            model_path="test.pt",
            conf=0.3,
            iou=0.5,
            device="cpu",
        )

        assert detector.model_path == "test.pt"
        assert detector.conf == 0.3
        assert detector.iou == 0.5
        assert detector.device == "cpu"
        assert detector._model is None

    def test_yolo_detector_is_loaded_false(self):
        """Test YOLOPlateDetector is_loaded when not loaded."""
        from src.models.plate_detector import YOLOPlateDetector

        detector = YOLOPlateDetector(model_path="test.pt")
        assert detector.is_loaded is False

    def test_yolo_detector_load(self):
        """Test YOLOPlateDetector load."""
        from src.models.plate_detector import YOLOPlateDetector

        with patch('src.models.plate_detector.YOLO') as mock_yolo:
            detector = YOLOPlateDetector(model_path="test.pt")
            detector.load()

            assert detector.is_loaded is True
            mock_yolo.assert_called_once_with("test.pt")

    def test_yolo_detector_detect_not_loaded(self):
        """Test YOLOPlateDetector detect raises when not loaded."""
        from src.models.plate_detector import YOLOPlateDetector

        detector = YOLOPlateDetector(model_path="test.pt")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="not loaded"):
            detector.detect(frame)

    def test_yolo_detector_detect_no_results(self):
        """Test YOLOPlateDetector detect with no detections."""
        from src.models.plate_detector import YOLOPlateDetector

        with patch('src.models.plate_detector.YOLO') as mock_yolo, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_model = MagicMock()
            mock_model.predict.return_value = []
            mock_yolo.return_value = mock_model

            detector = YOLOPlateDetector(model_path="test.pt")
            detector.load()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = detector.detect(frame)

            assert results == []

    def test_yolo_detector_detect_empty_boxes(self):
        """Test YOLOPlateDetector detect with empty boxes."""
        from src.models.plate_detector import YOLOPlateDetector

        with patch('src.models.plate_detector.YOLO') as mock_yolo, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_model = MagicMock()
            mock_model.names = {0: "plate"}
            mock_model.predict.return_value = [MagicMock(boxes=None)]
            mock_yolo.return_value = mock_model

            detector = YOLOPlateDetector(model_path="test.pt")
            detector.load()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = detector.detect(frame)

            assert results == []

    def test_create_plate_detector(self):
        """Test create_plate_detector factory function."""
        from src.models.plate_detector import create_plate_detector

        with patch('src.models.plate_detector.YOLO') as mock_yolo:
            mock_yolo.return_value = MagicMock()

            detector = create_plate_detector(
                model_path="test.pt",
                conf=0.3,
                iou=0.5,
                device="cpu",
            )

            assert detector is not None
            assert detector.conf == 0.3


class TestTextDetectorModels:
    """Test text detector models."""

    def test_paddle_detector_init(self):
        """Test PaddleTextDetector initialization."""
        from src.models.text_detector import PaddleTextDetector

        detector = PaddleTextDetector(
            model_dir="./models",
            device="cpu",
        )

        assert detector.model_dir == "./models"
        assert detector.device == "cpu"
        assert detector._model is None

    def test_paddle_detector_is_loaded_false(self):
        """Test PaddleTextDetector is_loaded when not loaded."""
        from src.models.text_detector import PaddleTextDetector

        detector = PaddleTextDetector()
        assert detector.is_loaded is False

    def test_paddle_detector_load(self):
        """Test PaddleTextDetector load."""
        from src.models.text_detector import PaddleTextDetector

        with patch('src.models.text_detector.TextDetection') as mock_text_det:
            mock_text_det.return_value = MagicMock()

            detector = PaddleTextDetector(model_dir="./models", device="cpu")
            detector.load()

            assert detector.is_loaded is True

    def test_paddle_detector_detect_not_loaded(self):
        """Test PaddleTextDetector detect raises when not loaded."""
        from src.models.text_detector import PaddleTextDetector

        detector = PaddleTextDetector()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="not loaded"):
            detector.detect(frame)

    def test_paddle_detector_detect_no_results(self):
        """Test PaddleTextDetector detect with no detections."""
        from src.models.text_detector import PaddleTextDetector

        with patch('src.models.text_detector.TextDetection') as mock_text_det, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_model = MagicMock()
            mock_model.predict.return_value = []
            mock_text_det.return_value = mock_model

            detector = PaddleTextDetector(model_dir="./models", device="cpu")
            detector.load()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = detector.detect(frame)

            assert results == []

    def test_paddle_detector_detect_empty_polys(self):
        """Test PaddleTextDetector detect with empty polygons."""
        from src.models.text_detector import PaddleTextDetector

        with patch('src.models.text_detector.TextDetection') as mock_text_det, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_result = MagicMock()
            mock_result.json = lambda: {
                "res": {
                    "dt_polys": [],
                    "dt_scores": [],
                }
            }

            mock_model = MagicMock()
            mock_model.predict.return_value = [mock_result]
            mock_text_det.return_value = mock_model

            detector = PaddleTextDetector(model_dir="./models", device="cpu")
            detector.load()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = detector.detect(frame)

            assert results == []

    def test_parse_result_with_dict(self):
        """Test _parse_result with dict input."""
        from src.models.text_detector import _parse_result

        res = MagicMock()
        res.json = {"res": {"dt_polys": [], "dt_scores": []}}

        result = _parse_result(res)
        assert "dt_polys" in result

    def test_parse_result_with_callable(self):
        """Test _parse_result with callable json."""
        from src.models.text_detector import _parse_result

        res = MagicMock()
        res.json = lambda: {"res": {"dt_polys": [], "dt_scores": []}}

        result = _parse_result(res)
        assert "dt_polys" in result

    def test_parse_result_with_string(self):
        """Test _parse_result with string json."""
        import json

        from src.models.text_detector import _parse_result

        res = MagicMock()
        res.json = json.dumps({"res": {"dt_polys": [], "dt_scores": []}})

        result = _parse_result(res)
        assert "dt_polys" in result


class TestTextRecognizerModels:
    """Test text recognizer models."""

    def test_paddle_recognizer_init(self):
        """Test PaddleTextRecognizer initialization."""
        from src.models.text_recognizer import PaddleTextRecognizer

        recognizer = PaddleTextRecognizer(
            model_dir="./models",
            device="cpu",
        )

        assert recognizer.model_dir == "./models"
        assert recognizer.device == "cpu"
        assert recognizer._model is None

    def test_paddle_recognizer_is_loaded_false(self):
        """Test PaddleTextRecognizer is_loaded when not loaded."""
        from src.models.text_recognizer import PaddleTextRecognizer

        recognizer = PaddleTextRecognizer()
        assert recognizer.is_loaded is False

    def test_paddle_recognizer_load(self):
        """Test PaddleTextRecognizer load."""
        from src.models.text_recognizer import PaddleTextRecognizer

        with patch('src.models.text_recognizer.TextRecognition') as mock_rec:
            mock_rec.return_value = MagicMock()

            recognizer = PaddleTextRecognizer(model_dir="./models", device="cpu")
            recognizer.load()

            assert recognizer.is_loaded is True

    def test_paddle_recognizer_recognize_not_loaded(self):
        """Test PaddleTextRecognizer recognize raises when not loaded."""
        from src.models.text_recognizer import PaddleTextRecognizer

        recognizer = PaddleTextRecognizer()

        images = [np.zeros((50, 100, 3), dtype=np.uint8)]

        with pytest.raises(RuntimeError, match="not loaded"):
            recognizer.recognize(images)

    def test_paddle_recognizer_recognize_empty_images(self):
        """Test PaddleTextRecognizer recognize with empty images."""
        from src.models.text_recognizer import PaddleTextRecognizer

        with patch('src.models.text_recognizer.TextRecognition') as mock_rec:
            mock_model = MagicMock()
            mock_rec.return_value = mock_model

            recognizer = PaddleTextRecognizer(model_dir="./models", device="cpu")
            recognizer.load()

            results = recognizer.recognize([])

            assert results == []

    def test_paddle_recognizer_recognize_empty_results(self):
        """Test PaddleTextRecognizer recognize with no results."""
        from src.models.text_recognizer import PaddleTextRecognizer

        with patch('src.models.text_recognizer.TextRecognition') as mock_rec, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_model = MagicMock()
            mock_model.predict.return_value = []
            mock_rec.return_value = mock_model

            recognizer = PaddleTextRecognizer(model_dir="./models", device="cpu")
            recognizer.load()

            images = [np.zeros((50, 100, 3), dtype=np.uint8)]
            results = recognizer.recognize(images)

            assert results == []

    def test_paddle_recognizer_recognize_with_text(self):
        """Test PaddleTextRecognizer recognize with text."""
        from src.models.text_recognizer import PaddleTextRecognizer

        with patch('src.models.text_recognizer.TextRecognition') as mock_rec, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_result = MagicMock()
            mock_result.json = lambda: {
                "rec_text": "ABC123",
                "rec_score": 0.95,
            }

            mock_model = MagicMock()
            mock_model.predict.return_value = [mock_result]
            mock_rec.return_value = mock_model

            recognizer = PaddleTextRecognizer(model_dir="./models", device="cpu")
            recognizer.load()

            images = [np.zeros((50, 100, 3), dtype=np.uint8)]
            results = recognizer.recognize(images)

            assert len(results) == 1
            assert results[0].text == "ABC123"

    def test_paddle_recognizer_recognize_below_min_score(self):
        """Test PaddleTextRecognizer ignores results below min_score."""
        from src.models.text_recognizer import PaddleTextRecognizer

        with patch('src.models.text_recognizer.TextRecognition') as mock_rec, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_result = MagicMock()
            mock_result.json = lambda: {
                "rec_text": "ABC123",
                "rec_score": 0.3,
            }

            mock_model = MagicMock()
            mock_model.predict.return_value = [mock_result]
            mock_rec.return_value = mock_model

            recognizer = PaddleTextRecognizer(model_dir="./models", device="cpu")
            recognizer.load()

            images = [np.zeros((50, 100, 3), dtype=np.uint8)]
            results = recognizer.recognize(images, min_score=0.5)

            assert len(results) == 0

    def test_paddle_recognizer_recognize_empty_text(self):
        """Test PaddleTextRecognizer ignores results with empty text."""
        from src.models.text_recognizer import PaddleTextRecognizer

        with patch('src.models.text_recognizer.TextRecognition') as mock_rec, \
             patch('src.observability.profiler.get_profiler') as mock_profiler:
            mock_profiler.return_value.enabled = False

            mock_result = MagicMock()
            mock_result.json = lambda: {
                "rec_text": "   ",
                "rec_score": 0.95,
            }

            mock_model = MagicMock()
            mock_model.predict.return_value = [mock_result]
            mock_rec.return_value = mock_model

            recognizer = PaddleTextRecognizer(model_dir="./models", device="cpu")
            recognizer.load()

            images = [np.zeros((50, 100, 3), dtype=np.uint8)]
            results = recognizer.recognize(images)

            assert len(results) == 0

    def test_parse_result_with_dict(self):
        """Test _parse_result with dict input."""
        from src.models.text_recognizer import _parse_result

        res = MagicMock()
        res.json = {"rec_text": "ABC", "rec_score": 0.95}

        result = _parse_result(res)
        assert result["rec_text"] == "ABC"

    def test_parse_result_with_callable(self):
        """Test _parse_result with callable json."""
        from src.models.text_recognizer import _parse_result

        res = MagicMock()
        res.json = lambda: {"rec_text": "ABC", "rec_score": 0.95}

        result = _parse_result(res)
        assert result["rec_text"] == "ABC"

    def test_parse_result_with_string(self):
        """Test _parse_result with string json."""
        import json

        from src.models.text_recognizer import _parse_result

        res = MagicMock()
        res.json = json.dumps({"rec_text": "ABC", "rec_score": 0.95})

        result = _parse_result(res)
        assert result["rec_text"] == "ABC"
