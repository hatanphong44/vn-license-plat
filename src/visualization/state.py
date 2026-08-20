"""Visualization state management."""

from dataclasses import dataclass, field

import numpy as np

from src.domain.models import LPRResult


@dataclass
class VisualizationState:
    """State container for visualization."""

    current_frame: np.ndarray | None = None
    results: list[LPRResult] = field(default_factory=list)
    fps: float = 0.0
    frame_count: int = 0
    last_update: float = 0.0

    def update(
        self,
        frame: np.ndarray,
        results: list[LPRResult],
        fps: float = 0.0,
    ) -> None:
        """Update state with new frame and results.

        Args:
            frame: Current frame
            results: Current results
            fps: Current FPS
        """
        import time

        self.current_frame = frame
        self.results = results
        self.fps = fps
        self.frame_count += 1
        self.last_update = time.time()

    def clear(self) -> None:
        """Clear state."""
        self.current_frame = None
        self.results = []
        self.fps = 0.0

    def has_frame(self) -> bool:
        """Check if frame is available."""
        return self.current_frame is not None

    def has_results(self) -> bool:
        """Check if results are available."""
        return len(self.results) > 0
