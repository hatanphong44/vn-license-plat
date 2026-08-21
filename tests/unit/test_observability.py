"""Tests for src/observability/* modules"""

import time


class TestModelStats:
    """Test ModelStats dataclass."""

    def test_init(self):
        """Test ModelStats initialization."""
        from src.observability.profiler import ModelStats

        stats = ModelStats("Test Model")
        assert stats.name == "Test Model"
        assert stats.calls == 0
        assert stats.total_ms == 0.0
        assert stats.min_ms == float('inf')
        assert stats.max_ms == 0.0
        assert len(stats.recent) == 0

    def test_record(self):
        """Test ModelStats.record."""
        from src.observability.profiler import ModelStats

        stats = ModelStats("Test Model")
        stats.record(10.0)
        stats.record(20.0)
        stats.record(30.0)

        assert stats.calls == 3
        assert stats.total_ms == 60.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 30.0
        assert len(stats.recent) == 3

    def test_avg_ms(self):
        """Test ModelStats.avg_ms property."""
        from src.observability.profiler import ModelStats

        stats = ModelStats("Test Model")
        stats.record(10.0)
        stats.record(20.0)
        stats.record(30.0)

        assert stats.avg_ms == 20.0

    def test_avg_ms_zero_calls(self):
        """Test ModelStats.avg_ms with zero calls."""
        from src.observability.profiler import ModelStats

        stats = ModelStats("Test Model")
        assert stats.avg_ms == 0.0

    def test_p95_ms(self):
        """Test ModelStats.p95_ms property."""
        from src.observability.profiler import ModelStats

        stats = ModelStats("Test Model")
        for i in range(20):
            stats.record(float(i + 1))

        assert stats.p95_ms > 0


class TestRuntimeStats:
    """Test RuntimeStats class."""

    def test_init(self):
        """Test RuntimeStats initialization."""
        from src.observability.profiler import RuntimeStats

        stats = RuntimeStats()

        assert stats.camera_frames == 0
        assert stats.camera_errors == 0
        assert stats.inference_cycles == 0
        assert stats.pipeline_calls == 0
        assert stats.windows_completed == 0
        assert stats.events_published == 0


class TestDebugProfiler:
    """Test DebugProfiler class."""

    def test_init_disabled(self):
        """Test DebugProfiler initialization as disabled."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=False)
        assert profiler.enabled is False

    def test_init_enabled(self):
        """Test DebugProfiler initialization as enabled."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        assert profiler.enabled is True

    def test_camera_read_disabled(self):
        """Test camera_read when disabled does nothing."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=False)
        profiler.camera_read()
        assert profiler._stats.camera_frames == 0

    def test_camera_read_enabled(self):
        """Test camera_read when enabled increments counter."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.camera_read()
        profiler.camera_read()
        assert profiler._stats.camera_frames == 2

    def test_camera_error_enabled(self):
        """Test camera_error increments counter."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.camera_error()
        assert profiler._stats.camera_errors == 1

    def test_inference_start(self):
        """Test inference_start records start time."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.inference_start()
        assert profiler._current_inference_start > 0

    def test_yolo_call(self):
        """Test yolo_call records timing."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.yolo_call(10.0)

        assert profiler._stats.yolo.calls == 1
        assert profiler._stats.yolo.total_ms == 10.0

    def test_ocr_detection(self):
        """Test ocr_detection records timing."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.ocr_detection(15.0)

        assert profiler._stats.ocr_det.calls == 1
        assert profiler._stats.ocr_det.total_ms == 15.0

    def test_ocr_recognition(self):
        """Test ocr_recognition records timing."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.ocr_recognition(20.0)

        assert profiler._stats.ocr_rec.calls == 1
        assert profiler._stats.ocr_rec.total_ms == 20.0

    def test_pipeline_call(self):
        """Test pipeline_call records stats."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.pipeline_call(50.0, plates_found=3)

        assert profiler._stats.pipeline_calls == 1
        assert profiler._stats.pipeline_total_ms == 50.0
        assert profiler._stats.plates_detected == 3

    def test_window_finalized(self):
        """Test window_finalized increments counter."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.window_finalized(1)

        assert profiler._stats.windows_completed == 1

    def test_should_print_summary_not_time(self):
        """Test should_print_summary returns False when not time."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        assert profiler.should_print_summary() is False

    def test_should_print_summary_time_reached(self):
        """Test should_print_summary returns True when interval reached."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler._stats._last_summary_time = time.time() - 10.0  # 10 seconds ago
        assert profiler.should_print_summary() is True

    def test_should_print_summary_disabled(self):
        """Test should_print_summary returns False when disabled."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=False)
        assert profiler.should_print_summary() is False

    def test_find_bottleneck(self):
        """Test _find_bottleneck identifies slowest model."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.yolo_call(100.0)
        profiler.ocr_detection(50.0)
        profiler.ocr_recognition(200.0)

        bottleneck = profiler._find_bottleneck()
        assert "OCR recognition" in bottleneck

    def test_find_bottleneck_empty(self):
        """Test _find_bottleneck with no data."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        bottleneck = profiler._find_bottleneck()
        assert bottleneck == ""

    def test_reset(self):
        """Test reset clears all stats."""
        from src.observability.profiler import DebugProfiler

        profiler = DebugProfiler(enabled=True)
        profiler.yolo_call(10.0)
        profiler.camera_read()

        profiler.reset()

        assert profiler._stats.yolo.calls == 0
        assert profiler._stats.camera_frames == 0


class TestProfilerFunctions:
    """Test profiler module functions."""

    def test_get_profiler(self):
        """Test get_profiler returns global instance."""
        # Reset global
        import src.observability.profiler
        from src.observability import get_profiler
        src.observability.profiler._profiler = None

        profiler = get_profiler()
        assert profiler is not None
        assert profiler.enabled is False

    def test_init_profiler(self):
        """Test init_profiler creates enabled profiler."""
        from src.observability import init_profiler

        profiler = init_profiler(enabled=True)
        assert profiler is not None
        assert profiler.enabled is True

    def test_set_profiler_enabled(self):
        """Test set_profiler_enabled toggles profiler."""
        # Initialize
        import src.observability.profiler
        from src.observability import get_profiler, set_profiler_enabled
        src.observability.profiler._profiler = None
        get_profiler()

        set_profiler_enabled(True)
        assert get_profiler().enabled is True

        set_profiler_enabled(False)
        assert get_profiler().enabled is False


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_init(self):
        """Test MetricsCollector initialization."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        assert collector._metrics.frames_processed == 0
        assert collector._metrics.plates_detected == 0

    def test_record_frame(self):
        """Test record_frame increments counters."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_frame(10.0)

        assert collector._metrics.frames_processed == 1
        assert collector._metrics.last_inference_time == 10.0

    def test_record_frame_calculates_avg(self):
        """Test record_frame calculates average inference time."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_frame(10.0)
        collector.record_frame(20.0)

        assert collector._metrics.avg_inference_time == 15.0

    def test_record_plate(self):
        """Test record_plate increments counters."""
        from src.domain.models import LPRResult
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()

        result = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 10, 100, 50],
            yolo_score=0.95,
            class_name="plate",
            ocr_results=[],
        )

        collector.record_plate(result)

        assert collector._metrics.plates_detected == 1
        assert collector._metrics.unique_plates == 1
        assert collector._metrics.last_plate == "ABC123"

    def test_record_plate_counts_unique(self):
        """Test record_plate counts unique plates."""
        from src.domain.models import LPRResult
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()

        result1 = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 10, 100, 50],
            yolo_score=0.95,
            class_name="plate",
            ocr_results=[],
        )
        result2 = LPRResult(
            plate_index=1,
            plate="XYZ789",
            plate_normalized="XYZ789",
            box=[200, 200, 300, 250],
            yolo_score=0.90,
            class_name="plate",
            ocr_results=[],
        )

        collector.record_plate(result1)
        collector.record_plate(result2)
        collector.record_plate(result1)  # Duplicate

        assert collector._metrics.unique_plates == 2

    def test_record_event_success(self):
        """Test record_event with success."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_event(success=True)

        assert collector._metrics.events_published == 1
        assert collector._metrics.events_failed == 0

    def test_record_event_failure(self):
        """Test record_event with failure."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_event(success=False)

        assert collector._metrics.events_published == 0
        assert collector._metrics.events_failed == 1

    def test_record_camera_error(self):
        """Test record_camera_error increments counter."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_camera_error()

        assert collector._metrics.camera_errors == 1

    def test_record_inference_error(self):
        """Test record_inference_error increments counter."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_inference_error()

        assert collector._metrics.inference_errors == 1

    def test_record_publish_error(self):
        """Test record_publish_error increments counter."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_publish_error()

        assert collector._metrics.publish_errors == 1

    def test_get_metrics(self):
        """Test get_metrics returns dict."""
        from src.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        metrics = collector.get_metrics()

        assert isinstance(metrics, dict)
        assert "frames_processed" in metrics
        assert "plates_detected" in metrics

    def test_get_runtime_metrics(self):
        """Test get_runtime_metrics returns RuntimeMetrics."""
        from src.observability.metrics import MetricsCollector, RuntimeMetrics

        collector = MetricsCollector()
        metrics = collector.get_runtime_metrics()

        assert isinstance(metrics, RuntimeMetrics)


class TestRuntimeMetrics:
    """Test RuntimeMetrics dataclass."""

    def test_init(self):
        """Test RuntimeMetrics initialization."""
        from src.observability.metrics import RuntimeMetrics

        metrics = RuntimeMetrics()
        assert metrics.frames_processed == 0
        assert metrics.plates_detected == 0

    def test_get_fps(self):
        """Test get_fps returns current_fps."""
        from src.observability.metrics import RuntimeMetrics

        metrics = RuntimeMetrics()
        metrics.current_fps = 30.5

        assert metrics.get_fps() == 30.5

    def test_to_dict(self):
        """Test to_dict returns dictionary."""
        from src.observability.metrics import RuntimeMetrics

        metrics = RuntimeMetrics()
        metrics.frames_processed = 100
        metrics.current_fps = 30.5

        d = metrics.to_dict()

        assert d["frames_processed"] == 100
        assert d["current_fps"] == 30.5
