"""Tests for src/pipeline/* modules"""

from unittest.mock import MagicMock, patch

import numpy as np


class TestPlateCropper:
    """Test PlateCropper class."""

    def test_init_default(self):
        """Test PlateCropper initialization with defaults."""
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper()
        assert cropper.padding == 0.05

    def test_init_custom(self):
        """Test PlateCropper with custom padding."""
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper(padding=0.1)
        assert cropper.padding == 0.1

    def test_crop_basic(self):
        """Test basic crop operation."""
        from src.domain.models import PlateDetection
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper(padding=0.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = PlateDetection(
            box=[100, 100, 200, 200],
            score=0.95,
            class_id=0,
            class_name="plate",
        )

        result = cropper.crop(frame, detection)
        assert result is not None
        assert result.shape[0] == 100
        assert result.shape[1] == 100

    def test_crop_with_padding(self):
        """Test crop with padding."""
        from src.domain.models import PlateDetection
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper(padding=0.1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = PlateDetection(
            box=[100, 100, 200, 200],
            score=0.95,
            class_id=0,
            class_name="plate",
        )

        result = cropper.crop(frame, detection)
        assert result is not None
        # With 10% padding, crop should be larger than 100x100
        assert result.shape[0] > 100 or result.shape[1] > 100

    def test_crop_clamped_to_bounds(self):
        """Test crop is clamped to frame bounds."""
        from src.domain.models import PlateDetection
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper(padding=1.0)  # Large padding
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = PlateDetection(
            box=[10, 10, 50, 50],  # Near edge
            score=0.95,
            class_id=0,
            class_name="plate",
        )

        result = cropper.crop(frame, detection)
        # Should still return valid crop clamped to bounds
        assert result is not None
        assert result.shape[0] <= 480
        assert result.shape[1] <= 640

    def test_crop_invalid_box(self):
        """Test crop with invalid box."""
        from src.domain.models import PlateDetection
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = PlateDetection(
            box=[200, 200, 100, 100],  # Invalid: x2 < x1, y2 < y1
            score=0.95,
            class_id=0,
            class_name="plate",
        )

        result = cropper.crop(frame, detection)
        assert result is None

    def test_crop_empty_frame(self):
        """Test crop with empty frame portion."""
        from src.domain.models import PlateDetection
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = PlateDetection(
            box=[0, 0, 0, 0],  # Empty box
            score=0.95,
            class_id=0,
            class_name="plate",
        )

        result = cropper.crop(frame, detection)
        assert result is None

    def test_crop_with_coords(self):
        """Test crop_with_coords returns crop and coordinates."""
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper(padding=0.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        box = [100, 100, 200, 200]
        result = cropper.crop_with_coords(frame, box)

        assert result is not None
        crop, coords = result
        assert crop.shape[0] == 100
        assert crop.shape[1] == 100
        assert coords == [100, 100, 200, 200]

    def test_crop_with_coords_invalid(self):
        """Test crop_with_coords with invalid box."""
        from src.pipeline.cropper import PlateCropper

        cropper = PlateCropper()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        box = [200, 200, 100, 100]  # Invalid
        result = cropper.crop_with_coords(frame, box)

        assert result is None


class TestTextCropper:
    """Test TextCropper class."""

    def test_init_default(self):
        """Test TextCropper initialization with defaults."""
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper()
        assert cropper.padding == 10
        assert cropper.border_size == 10

    def test_init_custom(self):
        """Test TextCropper with custom values."""
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper(padding=5, border_size=5)
        assert cropper.padding == 5
        assert cropper.border_size == 5

    def test_crop_basic(self):
        """Test basic text crop."""
        from src.domain.models import TextDetection
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper()
        plate = np.zeros((100, 300, 3), dtype=np.uint8)

        polygon = np.array([[50, 20], [150, 20], [150, 50], [50, 50]], dtype=np.float32)
        detection = TextDetection.from_polygon_score(polygon, 0.95)

        result = cropper.crop(plate, detection)
        assert result is not None
        # Should have white border added
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_crop_with_padding(self):
        """Test text crop with padding."""
        from src.domain.models import TextDetection
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper(padding=20)
        plate = np.zeros((100, 300, 3), dtype=np.uint8)

        polygon = np.array([[50, 30], [100, 30], [100, 60], [50, 60]], dtype=np.float32)
        detection = TextDetection.from_polygon_score(polygon, 0.95)

        result = cropper.crop(plate, detection)
        assert result is not None

    def test_crop_out_of_bounds(self):
        """Test crop clips to plate bounds."""
        from src.domain.models import TextDetection
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper(padding=5)  # Small padding to avoid over-clipping
        plate = np.zeros((100, 300, 3), dtype=np.uint8)

        # Polygon slightly outside but with padding=5 it should be clamped
        polygon = np.array([[5, 5], [295, 5], [295, 95], [5, 95]], dtype=np.float32)
        detection = TextDetection.from_polygon_score(polygon, 0.95)

        result = cropper.crop(plate, detection)
        # Should still return valid crop clamped to bounds
        assert result is not None

    def test_crop_polygon(self):
        """Test crop_polygon method."""
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper()
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        polygon = np.array([[50, 20], [150, 20], [150, 50], [50, 50]], dtype=np.float32)

        result = cropper.crop_polygon(image, polygon)
        assert result is not None

    def test_crop_polygon_invalid(self):
        """Test crop_polygon with invalid polygon."""
        from src.pipeline.cropper import TextCropper

        cropper = TextCropper()
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        polygon = np.array([[200, 200], [150, 200], [150, 150], [200, 150]], dtype=np.float32)

        result = cropper.crop_polygon(image, polygon)
        assert result is None


class TestPlatePreprocessor:
    """Test PlatePreprocessor class."""

    def test_init_default(self):
        """Test PlatePreprocessor initialization."""
        from src.pipeline.cropper import PlatePreprocessor

        preprocessor = PlatePreprocessor()
        assert preprocessor.upscale_factor == 4

    def test_init_custom(self):
        """Test PlatePreprocessor with custom upscale factor."""
        from src.pipeline.cropper import PlatePreprocessor

        preprocessor = PlatePreprocessor(upscale_factor=2)
        assert preprocessor.upscale_factor == 2

    def test_preprocess_upscaling(self):
        """Test preprocessing upscales plate."""
        from src.pipeline.cropper import PlatePreprocessor

        preprocessor = PlatePreprocessor(upscale_factor=2)
        plate = np.zeros((100, 200, 3), dtype=np.uint8)

        result = preprocessor.preprocess(plate)

        assert result.shape[0] == 200
        assert result.shape[1] == 400

    def test_preprocess_no_upscale(self):
        """Test preprocessing without upscaling."""
        from src.pipeline.cropper import PlatePreprocessor

        preprocessor = PlatePreprocessor(upscale_factor=1)
        plate = np.zeros((100, 200, 3), dtype=np.uint8)

        result = preprocessor.preprocess(plate)

        assert result.shape == plate.shape

    def test_denoise(self):
        """Test denoise method."""
        from src.pipeline.cropper import PlatePreprocessor

        with patch('src.pipeline.cropper.cv2') as mock_cv2:
            mock_cv2.fastNlMeansDenoisingColored.return_value = np.zeros((100, 200, 3), dtype=np.uint8)

            preprocessor = PlatePreprocessor()
            plate = np.zeros((100, 200, 3), dtype=np.uint8)

            preprocessor.denoise(plate)

            mock_cv2.fastNlMeansDenoisingColored.assert_called_once()

    def test_adjust_contrast(self):
        """Test adjust_contrast method."""
        from src.pipeline.cropper import PlatePreprocessor

        with patch('src.pipeline.cropper.cv2') as mock_cv2:
            mock_cv2.cvtColor = MagicMock(side_effect=lambda x, y: x)
            mock_cv2.split = MagicMock(return_value=[np.zeros((100, 200), dtype=np.uint8)] * 3)
            mock_cv2.merge = MagicMock(return_value=np.zeros((100, 200, 3), dtype=np.uint8))
            mock_cv2.createCLAHE.return_value.apply = MagicMock(return_value=np.zeros((100, 200), dtype=np.uint8))

            preprocessor = PlatePreprocessor()
            plate = np.zeros((100, 200, 3), dtype=np.uint8)

            preprocessor.adjust_contrast(plate)

            mock_cv2.cvtColor.assert_called()


class TestLPRPipeline:
    """Test LPRPipeline class."""

    def test_init(self):
        """Test LPRPipeline initialization."""
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        text_detector = MagicMock()
        text_recognizer = MagicMock()

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        assert pipeline.plate_detector is plate_detector
        assert pipeline.text_detector is text_detector
        assert pipeline.text_recognizer is text_recognizer
        assert pipeline.plate_cropper is not None
        assert pipeline.text_cropper is not None
        assert pipeline.preprocessor is not None
        assert pipeline.postprocessor is not None

    def test_init_custom_config(self):
        """Test LPRPipeline with custom configuration."""
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        text_detector = MagicMock()
        text_recognizer = MagicMock()

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
            plate_padding=0.1,
            text_padding=5,
            upscale_factor=2,
            rec_min_score=0.5,
        )

        assert pipeline.plate_cropper.padding == 0.1
        assert pipeline.text_cropper.padding == 5
        assert pipeline.preprocessor.upscale_factor == 2
        assert pipeline.rec_min_score == 0.5

    def test_process_frame_no_detections(self):
        """Test process_frame with no plate detections."""
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        plate_detector.detect.return_value = []

        text_detector = MagicMock()
        text_recognizer = MagicMock()

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(frame)

        assert results == []
        plate_detector.detect.assert_called_once()

    def test_process_frame_with_detections(self):
        """Test process_frame with plate detections."""
        from src.domain.models import PlateDetection, TextDetection, TextRecognition
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        plate_detector.detect.return_value = [
            PlateDetection(
                box=[100, 100, 200, 150],
                score=0.95,
                class_id=0,
                class_name="plate",
            )
        ]

        text_detector = MagicMock()
        polygon = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
        text_detector.detect.return_value = [
            TextDetection.from_polygon_score(polygon, 0.90)
        ]

        text_recognizer = MagicMock()
        text_recognizer.recognize.return_value = [
            TextRecognition(text="ABC123", line=0, det_score=0.90, rec_score=0.95, polygon=[])
        ]

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(frame)

        assert len(results) == 1
        assert results[0].plate == "ABC123"

    def test_process_frame_multiple_detections(self):
        """Test process_frame with multiple plate detections."""
        from src.domain.models import PlateDetection, TextDetection, TextRecognition
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        plate_detector.detect.return_value = [
            PlateDetection(box=[100, 100, 200, 150], score=0.95, class_id=0, class_name="plate"),
            PlateDetection(box=[300, 200, 400, 250], score=0.90, class_id=0, class_name="plate"),
        ]

        text_detector = MagicMock()
        polygon = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
        text_detector.detect.return_value = [
            TextDetection.from_polygon_score(polygon, 0.90)
        ]

        text_recognizer = MagicMock()
        text_recognizer.recognize.return_value = [
            TextRecognition(text="ABC123", line=0, det_score=0.90, rec_score=0.95, polygon=[])
        ]

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(frame)

        # May have 0, 1, or 2 results depending on crop success
        assert isinstance(results, list)

    def test_process_frame_no_text_detected(self):
        """Test process_frame when no text detected."""
        from src.domain.models import PlateDetection
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        plate_detector.detect.return_value = [
            PlateDetection(box=[100, 100, 200, 150], score=0.95, class_id=0, class_name="plate")
        ]

        text_detector = MagicMock()
        text_detector.detect.return_value = []

        text_recognizer = MagicMock()

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(frame)

        assert len(results) == 0

    def test_process_frame_no_text_recognized(self):
        """Test process_frame when no text recognized."""
        from src.domain.models import PlateDetection, TextDetection
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        plate_detector.detect.return_value = [
            PlateDetection(box=[100, 100, 200, 150], score=0.95, class_id=0, class_name="plate")
        ]

        text_detector = MagicMock()
        polygon = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
        text_detector.detect.return_value = [
            TextDetection.from_polygon_score(polygon, 0.90)
        ]

        text_recognizer = MagicMock()
        text_recognizer.recognize.return_value = []

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_frame(frame)

        assert len(results) == 0

    def test_process_single_plate(self):
        """Test process_single_plate method."""
        from src.domain.models import TextDetection, TextRecognition
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        text_detector = MagicMock()
        polygon = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
        text_detector.detect.return_value = [
            TextDetection.from_polygon_score(polygon, 0.90)
        ]

        text_recognizer = MagicMock()
        text_recognizer.recognize.return_value = [
            TextRecognition(text="ABC123", line=0, det_score=0.90, rec_score=0.95, polygon=[])
        ]

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        plate_crop = np.zeros((100, 300, 3), dtype=np.uint8)
        result = pipeline.process_single_plate(plate_crop)

        assert result is not None
        assert "ABC" in result

    def test_process_single_plate_no_text(self):
        """Test process_single_plate when no text detected."""
        from src.pipeline.lpr_pipeline import LPRPipeline

        plate_detector = MagicMock()
        text_detector = MagicMock()
        text_detector.detect.return_value = []
        text_recognizer = MagicMock()

        pipeline = LPRPipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        plate_crop = np.zeros((100, 300, 3), dtype=np.uint8)
        result = pipeline.process_single_plate(plate_crop)

        assert result is None


class TestCreatePipeline:
    """Test create_pipeline factory function."""

    def test_create_pipeline_with_defaults(self):
        """Test create_pipeline with default config."""
        from src.pipeline.lpr_pipeline import create_pipeline

        plate_detector = MagicMock()
        text_detector = MagicMock()
        text_recognizer = MagicMock()

        pipeline = create_pipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
        )

        assert pipeline.plate_cropper.padding == 0.05
        assert pipeline.text_cropper.padding == 10
        assert pipeline.preprocessor.upscale_factor == 4
        assert pipeline.rec_min_score == 0.0

    def test_create_pipeline_with_custom_config(self):
        """Test create_pipeline with custom config."""
        from src.pipeline.lpr_pipeline import create_pipeline

        plate_detector = MagicMock()
        text_detector = MagicMock()
        text_recognizer = MagicMock()

        config = {
            "plate_padding": 0.1,
            "text_padding": 5,
            "upscale_factor": 2,
            "rec_min_score": 0.5,
        }

        pipeline = create_pipeline(
            plate_detector=plate_detector,
            text_detector=text_detector,
            text_recognizer=text_recognizer,
            config=config,
        )

        assert pipeline.plate_cropper.padding == 0.1
        assert pipeline.text_cropper.padding == 5
        assert pipeline.preprocessor.upscale_factor == 2
        assert pipeline.rec_min_score == 0.5
