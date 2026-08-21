"""Tests for src/visualization/* modules"""

from unittest.mock import MagicMock, patch

import numpy as np


class TestResultAnnotator:
    """Test ResultAnnotator class."""

    def test_init_default_values(self):
        """Test ResultAnnotator initialization with defaults."""
        from src.visualization.annotator import ResultAnnotator

        annotator = ResultAnnotator()
        assert annotator.box_color == (0, 255, 0)
        assert annotator.text_color == (255, 255, 255)
        assert annotator.font_scale == 0.7
        assert annotator.thickness == 2

    def test_init_custom_values(self):
        """Test ResultAnnotator with custom values."""
        from src.visualization.annotator import ResultAnnotator

        annotator = ResultAnnotator(
            box_color=(255, 0, 0),
            text_color=(0, 0, 255),
            font_scale=1.0,
            thickness=3,
        )
        assert annotator.box_color == (255, 0, 0)
        assert annotator.text_color == (0, 0, 255)
        assert annotator.font_scale == 1.0
        assert annotator.thickness == 3

    def test_draw_result(self):
        """Test draw_result draws on frame."""
        from src.visualization.annotator import ResultAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.rectangle = MagicMock()
            mock_cv2.getTextSize = MagicMock(return_value=((100, 20), 5))
            mock_cv2.putText = MagicMock()

            annotator = ResultAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            mock_result = MagicMock()
            mock_result.box = [10, 10, 100, 50]
            mock_result.plate_normalized = "ABC123"
            mock_result.get_confidence.return_value = 0.95

            result = annotator.draw_result(frame, mock_result)

            assert result is not None
            mock_cv2.rectangle.assert_called()

    def test_draw_results_multiple(self):
        """Test draw_results draws multiple results."""
        from src.visualization.annotator import ResultAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.rectangle = MagicMock()
            mock_cv2.getTextSize = MagicMock(return_value=((100, 20), 5))
            mock_cv2.putText = MagicMock()

            annotator = ResultAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            mock_result1 = MagicMock()
            mock_result1.box = [10, 10, 100, 50]
            mock_result1.plate_normalized = "ABC123"
            mock_result1.get_confidence.return_value = 0.95

            mock_result2 = MagicMock()
            mock_result2.box = [200, 200, 300, 250]
            mock_result2.plate_normalized = "XYZ789"
            mock_result2.get_confidence.return_value = 0.88

            results = [mock_result1, mock_result2]
            annotated = annotator.draw_results(frame, results)

            assert annotated is not None
            assert mock_cv2.rectangle.call_count == 4  # 2 boxes + 2 backgrounds

    def test_draw_fps_top_left(self):
        """Test draw_fps draws FPS in top-left."""
        from src.visualization.annotator import ResultAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.getTextSize = MagicMock(return_value=((50, 20), 5))
            mock_cv2.rectangle = MagicMock()
            mock_cv2.putText = MagicMock()

            annotator = ResultAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            result = annotator.draw_fps(frame, 30.5, position="top-left")
            assert result is not None

    def test_draw_fps_top_right(self):
        """Test draw_fps draws FPS in top-right."""
        from src.visualization.annotator import ResultAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.getTextSize = MagicMock(return_value=((50, 20), 5))
            mock_cv2.rectangle = MagicMock()
            mock_cv2.putText = MagicMock()

            annotator = ResultAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            result = annotator.draw_fps(frame, 30.5, position="top-right")
            assert result is not None

    def test_draw_fps_bottom_left(self):
        """Test draw_fps draws FPS in bottom-left."""
        from src.visualization.annotator import ResultAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.getTextSize = MagicMock(return_value=((50, 20), 5))
            mock_cv2.rectangle = MagicMock()
            mock_cv2.putText = MagicMock()

            annotator = ResultAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            result = annotator.draw_fps(frame, 30.5, position="bottom-left")
            assert result is not None

    def test_draw_fps_bottom_right(self):
        """Test draw_fps draws FPS in bottom-right."""
        from src.visualization.annotator import ResultAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.getTextSize = MagicMock(return_value=((50, 20), 5))
            mock_cv2.rectangle = MagicMock()
            mock_cv2.putText = MagicMock()

            annotator = ResultAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            result = annotator.draw_fps(frame, 30.5, position="bottom-right")
            assert result is not None


class TestPlateDetectionAnnotator:
    """Test PlateDetectionAnnotator class."""

    def test_init(self):
        """Test PlateDetectionAnnotator initialization."""
        from src.visualization.annotator import PlateDetectionAnnotator

        annotator = PlateDetectionAnnotator()
        assert annotator.plate_color == (0, 255, 0)
        assert annotator.text_color_box == (255, 0, 0)
        assert annotator.rec_text_color == (0, 0, 255)

    def test_draw_plate_boxes(self):
        """Test draw_plate_boxes."""
        from src.visualization.annotator import PlateDetectionAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.rectangle = MagicMock()
            mock_cv2.putText = MagicMock()

            annotator = PlateDetectionAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            boxes = [[10, 10, 100, 50]]
            scores = [0.95]

            result = annotator.draw_plate_boxes(frame, boxes, scores)
            assert result is not None

    def test_draw_text_boxes(self):
        """Test draw_text_boxes."""
        from src.visualization.annotator import PlateDetectionAnnotator

        with patch('src.visualization.annotator.cv2') as mock_cv2:
            mock_cv2.polylines = MagicMock()
            mock_cv2.putText = MagicMock()

            annotator = PlateDetectionAnnotator()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            polygons = [np.array([[10, 10], [100, 10], [100, 50], [10, 50]])]
            texts = ["ABC123"]

            result = annotator.draw_text_boxes(frame, polygons, texts)
            assert result is not None


class TestVisualizationState:
    """Test VisualizationState class."""

    def test_init(self):
        """Test VisualizationState initialization."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        assert state.current_frame is None
        assert state.results == []
        assert state.fps == 0.0
        assert state.frame_count == 0

    def test_update(self):
        """Test state update."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [MagicMock()]

        state.update(frame, results, fps=30.5)

        assert state.current_frame is not None
        assert state.results == results
        assert state.fps == 30.5
        assert state.frame_count == 1
        assert state.last_update > 0

    def test_clear(self):
        """Test state clear."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state.update(frame, [MagicMock()], fps=30.5)

        state.clear()

        assert state.current_frame is None
        assert state.results == []
        assert state.fps == 0.0

    def test_has_frame_true(self):
        """Test has_frame returns True when frame exists."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state.update(frame, [], fps=0.0)

        assert state.has_frame() is True

    def test_has_frame_false(self):
        """Test has_frame returns False when no frame."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        assert state.has_frame() is False

    def test_has_results_true(self):
        """Test has_results returns True when results exist."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        state.update(np.zeros((100, 100, 3)), [MagicMock()], fps=0.0)

        assert state.has_results() is True

    def test_has_results_false(self):
        """Test has_results returns False when no results."""
        from src.visualization.state import VisualizationState

        state = VisualizationState()
        assert state.has_results() is False


class TestNoOpOverlayRenderer:
    """Test NoOpOverlayRenderer class."""

    def test_init(self):
        """Test NoOpOverlayRenderer initialization."""
        from src.visualization.overlay import NoOpOverlayRenderer

        renderer = NoOpOverlayRenderer()
        assert renderer is not None

    def test_start_stop(self):
        """Test NoOpOverlayRenderer start/stop."""
        from src.visualization.overlay import NoOpOverlayRenderer

        renderer = NoOpOverlayRenderer()
        renderer.start()  # Should do nothing
        renderer.stop()    # Should do nothing

    def test_update(self):
        """Test NoOpOverlayRenderer update."""
        from src.visualization.overlay import NoOpOverlayRenderer

        renderer = NoOpOverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [MagicMock()]

        renderer.update(frame, results, fps=30.5)  # Should do nothing


class TestHeadlessOverlayRenderer:
    """Test HeadlessOverlayRenderer class."""

    def test_init(self):
        """Test HeadlessOverlayRenderer initialization."""
        from src.visualization.overlay import HeadlessOverlayRenderer

        renderer = HeadlessOverlayRenderer()
        assert renderer.save_dir == "captures"
        assert renderer.save_interval_seconds == 5
        assert renderer._saved_count == 0

    def test_init_custom_values(self):
        """Test HeadlessOverlayRenderer with custom values."""
        from src.visualization.overlay import HeadlessOverlayRenderer

        renderer = HeadlessOverlayRenderer(
            save_dir="custom_dir",
            save_interval_seconds=10,
        )
        assert renderer.save_dir == "custom_dir"
        assert renderer.save_interval_seconds == 10

    def test_start_stop(self):
        """Test HeadlessOverlayRenderer start/stop."""
        from src.visualization.overlay import HeadlessOverlayRenderer

        with patch('os.makedirs'), patch('src.visualization.overlay.logger'):
            renderer = HeadlessOverlayRenderer()
            renderer.start()
            renderer.stop()

    def test_update_saves_frame(self):
        """Test HeadlessOverlayRenderer saves frames."""
        from src.visualization.overlay import HeadlessOverlayRenderer

        with patch('src.visualization.overlay.cv2') as mock_cv2, patch('os.makedirs'), \
             patch('time.time', return_value=0), patch('src.visualization.overlay.logger'):
            mock_cv2.imwrite = MagicMock()

            renderer = HeadlessOverlayRenderer(save_interval_seconds=0)
            renderer.start()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Use real LPRResult-like object with proper box attribute
            mock_result = MagicMock()
            mock_result.box = [10, 10, 100, 50]
            mock_result.plate_normalized = "ABC123"
            mock_result.get_confidence.return_value = 0.95

            renderer.update(frame, [mock_result], fps=30.5)

            mock_cv2.imwrite.assert_called_once()

    def test_update_no_save_when_no_interval(self):
        """Test HeadlessOverlayRenderer doesn't save when interval not reached."""
        from src.visualization.overlay import HeadlessOverlayRenderer

        with patch('src.visualization.overlay.cv2') as mock_cv2, patch('os.makedirs'), \
             patch('time.time', return_value=0), patch('src.visualization.overlay.logger'):
            renderer = HeadlessOverlayRenderer(save_interval_seconds=60)

            renderer._last_save_time = 1000  # Far in the past

            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Use real LPRResult-like object with proper box attribute
            mock_result = MagicMock()
            mock_result.box = [10, 10, 100, 50]
            mock_result.plate_normalized = "ABC123"
            mock_result.get_confidence.return_value = 0.95

            renderer.update(frame, [mock_result], fps=30.5)

            # Should not save because interval not reached
            mock_cv2.imwrite.assert_not_called()

    def test_push_frame(self):
        """Test HeadlessOverlayRenderer push_frame."""
        from src.visualization.overlay import HeadlessOverlayRenderer

        renderer = HeadlessOverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        renderer.push_frame(frame)  # Should do nothing


class TestOverlayRenderer:
    """Test OverlayRenderer class."""

    def test_init(self):
        """Test OverlayRenderer initialization."""
        from src.visualization.overlay import OverlayRenderer

        renderer = OverlayRenderer(
            display_fps=True,
            window_name="Test",
        )
        assert renderer.display_fps is True
        assert renderer.window_name == "Test"
        assert renderer._running is False

    def test_update(self):
        """Test OverlayRenderer update."""
        from src.visualization.overlay import OverlayRenderer

        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [MagicMock()]

        renderer.update(frame, results, fps=30.5)

        assert renderer._fps == 30.5
        assert renderer._results == results

    def test_start_stop(self):
        """Test OverlayRenderer start/stop."""
        from src.visualization.overlay import OverlayRenderer

        with patch('src.visualization.overlay.cv2') as mock_cv2, \
             patch('src.visualization.overlay.logger'):
            mock_cv2.namedWindow = MagicMock()
            mock_cv2.destroyAllWindows = MagicMock()

            renderer = OverlayRenderer()
            renderer.start()

            assert renderer._running is True

            renderer.stop()
            assert renderer._running is False

    def test_start_already_running(self):
        """Test OverlayRenderer start when already running."""
        from src.visualization.overlay import OverlayRenderer

        with patch('src.visualization.overlay.cv2') as mock_cv2, \
             patch('src.visualization.overlay.logger'):
            mock_cv2.namedWindow = MagicMock()
            mock_cv2.destroyAllWindows = MagicMock()

            renderer = OverlayRenderer()
            renderer.start()
            assert renderer._running is True

            # Try to start again
            renderer.start()
            assert renderer._running is True

            renderer.stop()

    def test_start_exception_handling(self):
        """Test OverlayRenderer handles exceptions during start."""
        from src.visualization.overlay import OverlayRenderer

        with patch('src.visualization.overlay.cv2') as mock_cv2, \
             patch('src.visualization.overlay.logger'):
            mock_cv2.namedWindow.side_effect = RuntimeError("Cannot create window")

            renderer = OverlayRenderer()
            renderer.start()

            # Should handle gracefully and not be running
            assert renderer._running is False

    def test_push_frame(self):
        """Test OverlayRenderer push_frame."""
        from src.visualization.overlay import OverlayRenderer

        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should not raise even when queue is full
        for _ in range(10):
            renderer.push_frame(frame)


class TestCreateOverlayRenderer:
    """Test create_overlay_renderer factory function."""

    def test_create_noop_when_disabled(self):
        """Test create_overlay_renderer returns NoOp when disabled."""
        from src.visualization.overlay import (
            NoOpOverlayRenderer,
            create_overlay_renderer,
        )

        renderer = create_overlay_renderer(enabled=False)
        assert isinstance(renderer, NoOpOverlayRenderer)

    def test_create_headless(self):
        """Test create_overlay_renderer returns HeadlessOverlayRenderer in headless mode."""
        from src.visualization.overlay import (
            HeadlessOverlayRenderer,
            create_overlay_renderer,
        )

        renderer = create_overlay_renderer(enabled=True, headless=True)
        assert isinstance(renderer, HeadlessOverlayRenderer)

    def test_create_with_defaults(self):
        """Test create_overlay_renderer with default values."""
        from src.visualization.overlay import create_overlay_renderer

        with patch('src.visualization.overlay.cv2') as mock_cv2, \
             patch('src.visualization.overlay.logger'):
            mock_cv2.namedWindow = MagicMock()
            mock_cv2.destroyAllWindows = MagicMock()

            renderer = create_overlay_renderer(enabled=True, headless=False)

            assert renderer is not None

    def test_create_fallback_to_headless_on_error(self):
        """Test create_overlay_renderer falls back to headless on error."""
        from src.visualization.overlay import (
            HeadlessOverlayRenderer,
            create_overlay_renderer,
        )

        with patch('src.visualization.overlay.cv2'), \
             patch('src.visualization.overlay.logger'), \
             patch('src.visualization.overlay.OverlayRenderer') as mock_renderer_class:
            mock_renderer_class.side_effect = Exception("No display")

            renderer = create_overlay_renderer(enabled=True, headless=False)

            assert isinstance(renderer, HeadlessOverlayRenderer)
