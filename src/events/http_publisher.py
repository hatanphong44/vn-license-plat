"""HTTP event publisher.

Publishes plate events to HTTP endpoint.
"""

import logging
import time

import requests

from src.domain.models import PlateEvent
from src.events.publisher import EventPublisher

logger = logging.getLogger("lpr.events.http_publisher")


class HTTPEventPublisher(EventPublisher):
    """HTTP-based event publisher."""

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        headers: dict | None = None,
    ):
        """Initialize HTTP publisher.

        Args:
            url: HTTP endpoint URL
            timeout: Request timeout in seconds
            retry_count: Number of retry attempts
            retry_delay: Delay between retries in seconds
            headers: Optional HTTP headers
        """
        self.url = url
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.headers = headers or {}

    def is_available(self) -> bool:
        """Check if endpoint is available."""
        return bool(self.url)

    def publish(self, event: PlateEvent) -> bool:
        """Publish event to HTTP endpoint.

        Args:
            event: Plate event to publish

        Returns:
            True if published successfully
        """
        if not self.is_available():
            logger.warning("Callback URL not configured")
            return False

        payload = event.to_dict()

        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout,
                    headers=self.headers,
                )
                response.raise_for_status()

                logger.info(f"Event published: plate={event.plate_normalized} "
                          f"status={response.status_code}")
                return True

            except requests.exceptions.Timeout:
                logger.warning(f"Publish timeout (attempt {attempt + 1}/{self.retry_count})")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.retry_count}): {e}")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                # Don't retry on 4xx errors
                if 400 <= e.response.status_code < 500:
                    return False
            except Exception as e:
                logger.error(f"Publish failed: plate={event.plate_normalized} error={e}")

            # Wait before retry
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)

        logger.error(f"Publish failed after {self.retry_count} attempts: "
                    f"plate={event.plate_normalized}")
        return False

    def publish_batch(self, events: list[PlateEvent]) -> list[bool]:
        """Publish multiple events.

        Args:
            events: List of events to publish

        Returns:
            List of success booleans
        """
        return [self.publish(event) for event in events]


class NoOpPublisher(EventPublisher):
    """No-op publisher for testing or when no endpoint is configured."""

    def __init__(self):
        self._published_count = 0

    def is_available(self) -> bool:
        return True

    def publish(self, event: PlateEvent) -> bool:
        """Log but don't actually publish."""
        self._published_count += 1
        logger.info(f"[NO-OP] Would publish: plate={event.plate_normalized} "
                   f"camera={event.camera} frames={event.frames_collected}")
        return True

    @property
    def published_count(self) -> int:
        """Number of events that would have been published."""
        return self._published_count


def create_http_publisher(
    url: str,
    timeout: float = 5.0,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> HTTPEventPublisher | NoOpPublisher:
    """Factory to create HTTP publisher.

    Args:
        url: HTTP endpoint URL (empty = NoOp publisher)
        timeout: Request timeout
        retry_count: Number of retries
        retry_delay: Delay between retries

    Returns:
        HTTP publisher or NoOp if URL is empty
    """
    if not url:
        logger.info("No callback URL configured, using NoOp publisher")
        return NoOpPublisher()

    return HTTPEventPublisher(
        url=url,
        timeout=timeout,
        retry_count=retry_count,
        retry_delay=retry_delay,
    )
