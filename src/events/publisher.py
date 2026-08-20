"""Event publisher base.

Responsibilities (per PLAN.md):
- Event creation, publishing
- Publisher must be replaceable by HTTP/Redis/Kafka later
"""

import abc

from src.domain.models import CapturedPlate, PlateEvent


class EventPublisher(abc.ABC):
    """Abstract base class for event publishers."""

    @abc.abstractmethod
    def publish(self, event: PlateEvent) -> bool:
        """Publish a plate event.

        Args:
            event: Plate event to publish

        Returns:
            True if published successfully
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if publisher is available."""
        ...

    def create_event(
        self,
        result: CapturedPlate,
        camera: str,
        frames_count: int,
    ) -> PlateEvent:
        """Create event from result.

        Args:
            result: Best LPR result
            camera: Camera identifier
            frames_count: Number of frames collected

        Returns:
            Plate event
        """
        return PlateEvent.from_result(result, camera, frames_count)
