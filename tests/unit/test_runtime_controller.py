"""Unit tests for runtime controller."""

from unittest.mock import MagicMock

from src.runtime.controller import RuntimeController, get_controller


class TestRuntimeController:
    """Test RuntimeController."""

    def test_init(self):
        """Test controller initialization."""
        controller = RuntimeController()
        assert controller.worker is None
        assert controller.is_running is False

    def test_start_worker(self):
        """Test starting a worker."""
        controller = RuntimeController()
        mock_worker = MagicMock()
        mock_worker.is_running = True

        controller.start(mock_worker)

        assert controller.worker is mock_worker
        mock_worker.start.assert_called_once()
        assert controller.is_running is True

    def test_start_already_running(self):
        """Test starting worker when already running."""
        controller = RuntimeController()
        mock_worker1 = MagicMock()
        mock_worker1.is_running = True
        mock_worker2 = MagicMock()
        mock_worker2.is_running = True

        controller.start(mock_worker1)
        controller.start(mock_worker2)

        # First worker should have been stopped
        mock_worker1.stop.assert_called_once()
        assert controller.worker is mock_worker2

    def test_stop_worker(self):
        """Test stopping a worker."""
        controller = RuntimeController()
        mock_worker = MagicMock()

        controller._worker = mock_worker
        controller.stop()

        mock_worker.stop.assert_called_once()
        assert controller.worker is None
        assert controller.is_running is False

    def test_stop_when_not_running(self):
        """Test stopping when no worker is running."""
        controller = RuntimeController()
        # Should not raise
        controller.stop()

    def test_get_stats_no_worker(self):
        """Test getting stats when no worker."""
        controller = RuntimeController()
        stats = controller.get_stats()
        assert stats == {"running": False}

    def test_get_stats_with_worker(self):
        """Test getting stats with worker."""
        controller = RuntimeController()
        mock_worker = MagicMock()
        mock_worker.is_running = True
        mock_worker.get_stats.return_value = {"running": True, "frames": 100}

        controller._worker = mock_worker
        stats = controller.get_stats()

        assert stats == {"running": True, "frames": 100}
        mock_worker.get_stats.assert_called_once()


class TestGetController:
    """Test get_controller function."""

    def test_get_controller_creates_instance(self):
        """Test that get_controller returns a controller."""
        controller = get_controller()
        assert isinstance(controller, RuntimeController)

    def test_get_controller_returns_same_instance(self):
        """Test that get_controller returns the same instance."""
        controller1 = get_controller()
        controller2 = get_controller()
        assert controller1 is controller2
