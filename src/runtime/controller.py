"""Runtime controller - Manages worker lifecycle."""

import logging
from typing import Optional

from src.runtime.worker import LPRRuntimeWorker, WorkerConfig


logger = logging.getLogger("lpr.runtime.controller")


class RuntimeController:
    """Controller for managing LPR runtime workers."""

    def __init__(self):
        self._worker: Optional[LPRRuntimeWorker] = None

    @property
    def worker(self) -> Optional[LPRRuntimeWorker]:
        """Get current worker."""
        return self._worker

    @property
    def is_running(self) -> bool:
        """Check if runtime is running."""
        return self._worker is not None and self._worker.is_running

    def start(self, worker: LPRRuntimeWorker) -> None:
        """Start a worker.

        Args:
            worker: Worker to start
        """
        if self.is_running:
            logger.warning("Runtime already running, stopping first...")
            self.stop()

        self._worker = worker
        self._worker.start()
        logger.info("Runtime started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop current worker.

        Args:
            timeout: Timeout in seconds
        """
        if self._worker is not None:
            self._worker.stop(timeout=timeout)
            self._worker = None
            logger.info("Runtime stopped")

    def get_stats(self) -> dict:
        """Get runtime stats."""
        if self._worker is None:
            return {"running": False}

        return self._worker.get_stats()


# Global controller instance
_controller: Optional[RuntimeController] = None


def get_controller() -> RuntimeController:
    """Get global runtime controller."""
    global _controller
    if _controller is None:
        _controller = RuntimeController()
    return _controller
