"""Tests for src/runtime/worker.py"""

import time
from unittest.mock import MagicMock, patch

import numpy as np


class TestPlateObservation:
    """Test PlateObservation dataclass."""

    def test_init(self):
        """Test PlateObservation initialization."""
        from src.runtime.worker import PlateObservation

        obs = PlateObservation(
            plate_normalized="ABC123",
            plate="ABC-123",
            confidence=0.95,
            yolo_score=0.90,
            box=[10, 10, 100, 50],
            ocr_results=[],
        )

        assert obs.plate_normalized == "ABC123"
        assert obs.plate == "ABC-123"
        assert obs.confidence == 0.95
        assert obs.yolo_score == 0.90
        assert obs.is_valid is True
        assert obs.timestamp > 0


class TestWorkerConfig:
    """Test WorkerConfig dataclass."""

    def test_init_defaults(self):
        """Test WorkerConfig with defaults."""
        from src.runtime.worker import WorkerConfig

        config = WorkerConfig()
        assert config.reconnect_delay == 3.0
        assert config.preview is False
        assert config.save_frames is False

    def test_init_custom(self):
        """Test WorkerConfig with custom values."""
        from src.runtime.worker import WorkerConfig

        config = WorkerConfig(
            reconnect_delay=5.0,
            preview=True,
            save_frames=True,
        )
        assert config.reconnect_delay == 5.0
        assert config.preview is True
        assert config.save_frames is True


class TestWindowResult:
    """Test WindowResult dataclass."""

    def test_init(self):
        """Test WindowResult initialization."""
        from src.runtime.worker import WindowResult

        result = WindowResult(
            window_id=1,
            duration=3.0,
            observations=30,
            valid_observations=25,
            invalid_observations=5,
            unique_plates=["ABC123"],
            candidate_counts={"ABC123": 25},
            result="ABC123",
            confidence=0.95,
            action="PUBLISH",
        )

        assert result.window_id == 1
        assert result.duration == 3.0
        assert result.result == "ABC123"
        assert result.action == "PUBLISH"


class TestLPRRuntimeWorker:
    """Test LPRRuntimeWorker class."""

    def test_init(self):
        """Test LPRRuntimeWorker initialization."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        assert worker.camera is camera
        assert worker.pipeline is pipeline
        assert worker.publisher is publisher
        assert worker.is_running is False
        assert worker._window_id == 0

    def test_init_with_config(self):
        """Test LPRRuntimeWorker with config."""
        from src.runtime.worker import LPRRuntimeWorker, WorkerConfig

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()
        config = WorkerConfig(reconnect_delay=5.0)

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
            config=config,
        )

        assert worker.config.reconnect_delay == 5.0

    def test_init_with_callbacks(self):
        """Test LPRRuntimeWorker with callbacks."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()
        on_frame = MagicMock()
        on_result = MagicMock()
        on_error = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
            on_frame=on_frame,
            on_result=on_result,
            on_error=on_error,
        )

        assert worker.on_frame is on_frame
        assert worker.on_result is on_result
        assert worker.on_error is on_error

    def test_is_running_property(self):
        """Test is_running property."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        assert worker.is_running is False
        worker._running = True
        assert worker.is_running is True

    def test_start(self):
        """Test worker start."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker.start()

        assert worker.is_running is True
        assert worker._stop_event.is_set() is False

        # Clean up
        worker.stop()

    def test_start_already_running(self):
        """Test worker start when already running."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._running = True

        # Should not start another thread
        worker.start()
        assert worker._thread is None

    def test_stop(self):
        """Test worker stop."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._running = True
        worker._thread = MagicMock()

        worker.stop()

        assert worker.is_running is False
        assert worker._stop_event.is_set() is True

    def test_stop_not_running(self):
        """Test worker stop when not running."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        # Should not raise
        worker.stop()

    def test_stop_with_overlay(self):
        """Test worker stop with overlay."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()
        overlay = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
            overlay=overlay,
        )

        worker._running = True
        worker._thread = MagicMock()

        worker.stop()

        overlay.stop.assert_called_once()

    def test_get_stats(self):
        """Test get_stats."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        camera.source = "test_camera"
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._running = True
        worker._current_fps = 30.5
        worker._window_id = 5

        stats = worker.get_stats()

        assert stats["running"] is True
        assert stats["camera"] == "test_camera"
        assert stats["current_fps"] == 30.5
        assert stats["window_id"] == 5

    def test_save_frame(self):
        """Test save_frame."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('os.makedirs'), patch('cv2.imwrite', return_value=True):
            filename = worker.save_frame(frame, "test")

            assert filename is not None
            assert "test_0000.jpg" in filename

    def test_save_frame_error(self):
        """Test save_frame handles error."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('os.makedirs', side_effect=Exception("Disk full")):
            result = worker.save_frame(frame, "test")
            assert result is None

    def test_wait_with_check(self):
        """Test _wait_with_check."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        start = time.time()
        worker._wait_with_check(0.1)
        elapsed = time.time() - start

        # Should take roughly 0.1 seconds
        assert 0.05 <= elapsed <= 0.3

    def test_wait_with_check_stops_early(self):
        """Test _wait_with_check stops early on stop event."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._stop_event.set()

        start = time.time()
        worker._wait_with_check(10.0)  # 10 seconds
        elapsed = time.time() - start

        # Should stop immediately due to stop event
        assert elapsed < 1.0


class TestWindowFinalization:
    """Test window finalization logic."""

    def test_finalize_no_observations(self):
        """Test finalize with no observations."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._window_observations = []
        worker._window_start = time.time() - 3.0

        profiler = MagicMock()
        worker._finalize_window(profiler)

        profiler.window_finalized.assert_called_once()

    def test_finalize_no_valid_observations(self):
        """Test finalize with only invalid observations."""
        from src.runtime.worker import LPRRuntimeWorker, PlateObservation

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        # Add invalid observation
        obs = PlateObservation(
            plate_normalized="ABC123",
            plate="ABC-123",
            confidence=0.95,
            yolo_score=0.90,
            box=[10, 10, 100, 50],
            ocr_results=[],
            is_valid=False,
        )
        worker._window_observations = [obs]
        worker._window_start = time.time() - 3.0

        profiler = MagicMock()
        worker._finalize_window(profiler)

        profiler.window_finalized.assert_called_once()

    def test_finalize_with_tie(self):
        """Test finalize handles tie between candidates."""
        from src.runtime.worker import LPRRuntimeWorker, PlateObservation

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        # Add equal count observations for different plates
        obs1 = PlateObservation(
            plate_normalized="ABC123",
            plate="ABC-123",
            confidence=0.95,
            yolo_score=0.90,
            box=[10, 10, 100, 50],
            ocr_results=[],
            is_valid=True,
        )
        obs2 = PlateObservation(
            plate_normalized="XYZ789",
            plate="XYZ-789",
            confidence=0.90,
            yolo_score=0.85,
            box=[200, 200, 300, 250],
            ocr_results=[],
            is_valid=True,
        )
        worker._window_observations = [obs1, obs2]
        worker._window_start = time.time() - 3.0

        profiler = MagicMock()
        worker._finalize_window(profiler)

        profiler.window_finalized.assert_called_once()

    def test_finalize_publish_new_plate(self):
        """Test finalize publishes new plate."""
        from src.runtime.worker import LPRRuntimeWorker, PlateObservation

        camera = MagicMock()
        camera.source = "test"
        pipeline = MagicMock()
        pipeline.postprocessor.normalizer.is_valid.return_value = True
        publisher = MagicMock()
        publisher.publish.return_value = True
        publisher.create_event.return_value = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        # Add multiple observations for same plate
        for _ in range(10):
            obs = PlateObservation(
                plate_normalized="ABC123",
                plate="ABC-123",
                confidence=0.95,
                yolo_score=0.90,
                box=[10, 10, 100, 50],
                ocr_results=[],
                is_valid=True,
            )
            worker._window_observations.append(obs)

        worker._window_start = time.time() - 3.0

        profiler = MagicMock()
        worker._finalize_window(profiler)

        profiler.window_finalized.assert_called_once()

    def test_finalize_skip_duplicate(self):
        """Test finalize skips duplicate plate."""
        from src.runtime.worker import LPRRuntimeWorker, PlateObservation

        camera = MagicMock()
        camera.source = "test"
        pipeline = MagicMock()
        pipeline.postprocessor.normalizer.is_valid.return_value = True
        publisher = MagicMock()
        publisher.publish.return_value = True
        publisher.create_event.return_value = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        # Set last published to same plate
        worker._last_published_plate = "ABC123"

        # Add observations
        obs = PlateObservation(
            plate_normalized="ABC123",
            plate="ABC-123",
            confidence=0.95,
            yolo_score=0.90,
            box=[10, 10, 100, 50],
            ocr_results=[],
            is_valid=True,
        )
        worker._window_observations = [obs]
        worker._window_start = time.time() - 3.0

        profiler = MagicMock()
        worker._finalize_window(profiler)

        # Should not publish
        publisher.publish.assert_not_called()


class TestPublishPlate:
    """Test _publish_plate method."""

    def test_publish_success(self):
        """Test successful publish."""
        from src.runtime.worker import LPRRuntimeWorker, PlateObservation

        camera = MagicMock()
        camera.source = "test"
        pipeline = MagicMock()
        publisher = MagicMock()
        publisher.publish.return_value = True
        publisher.create_event.return_value = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        obs = PlateObservation(
            plate_normalized="ABC123",
            plate="ABC-123",
            confidence=0.95,
            yolo_score=0.90,
            box=[10, 10, 100, 50],
            ocr_results=[],
            is_valid=True,
        )
        worker._window_observations = [obs]

        profiler = MagicMock()
        result = worker._publish_plate("ABC123", 0.95, 1, 3.0, profiler)

        assert result is True
        publisher.publish.assert_called_once()

    def test_publish_no_matching_observation(self):
        """Test publish with no matching observation."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._window_observations = []

        profiler = MagicMock()
        result = worker._publish_plate("ABC123", 0.95, 1, 3.0, profiler)

        assert result is False


class TestCheckWindowCompletion:
    """Test _check_window_completion method."""

    def test_not_complete_yet(self):
        """Test window not complete yet."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        worker._window_start = time.time()  # Just started
        worker._window_id = 0

        profiler = MagicMock()
        worker._check_window_completion(profiler)

        # Window should not be finalized
        profiler.window_finalized.assert_not_called()

    def test_complete_and_new_window(self):
        """Test window completes and new window starts."""
        from src.runtime.worker import LPRRuntimeWorker

        camera = MagicMock()
        pipeline = MagicMock()
        publisher = MagicMock()

        worker = LPRRuntimeWorker(
            camera=camera,
            pipeline=pipeline,
            publisher=publisher,
        )

        original_id = worker._window_id
        worker._window_start = time.time() - 4.0  # 4 seconds ago
        worker._window_id = original_id

        profiler = MagicMock()
        worker._check_window_completion(profiler)

        # Window should be finalized and new started
        profiler.window_finalized.assert_called_once()
        assert worker._window_id == original_id + 1
