"""Unit tests for HTTP event publisher."""

from unittest.mock import MagicMock, patch

from src.domain.models import PlateEvent
from src.events.http_publisher import (
    HTTPEventPublisher,
    NoOpPublisher,
    create_http_publisher,
)


class TestNoOpPublisher:
    """Test NoOp publisher."""

    def test_is_available(self):
        """NoOp publisher should always be available."""
        publisher = NoOpPublisher()
        assert publisher.is_available() is True

    def test_publish_always_returns_true(self):
        """NoOp publisher should always return True."""
        publisher = NoOpPublisher()
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )
        assert publisher.publish(event) is True

    def test_publish_increments_count(self):
        """NoOp publisher should track publish count."""
        publisher = NoOpPublisher()
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        assert publisher.published_count == 0
        publisher.publish(event)
        assert publisher.published_count == 1
        publisher.publish(event)
        publisher.publish(event)
        assert publisher.published_count == 3


class TestHTTPEventPublisher:
    """Test HTTP event publisher."""

    def test_init_defaults(self):
        """Test default initialization."""
        publisher = HTTPEventPublisher(url="http://example.com")
        assert publisher.url == "http://example.com"
        assert publisher.timeout == 5.0
        assert publisher.retry_count == 3
        assert publisher.retry_delay == 1.0
        assert publisher.headers == {}

    def test_init_with_custom_values(self):
        """Test custom initialization."""
        publisher = HTTPEventPublisher(
            url="http://example.com",
            timeout=10.0,
            retry_count=5,
            retry_delay=2.0,
            headers={"Authorization": "Bearer token"},
        )
        assert publisher.timeout == 10.0
        assert publisher.retry_count == 5
        assert publisher.retry_delay == 2.0
        assert publisher.headers == {"Authorization": "Bearer token"}

    def test_is_available_with_url(self):
        """HTTP publisher should be available when URL is set."""
        publisher = HTTPEventPublisher(url="http://example.com")
        assert publisher.is_available() is True

    def test_is_available_without_url(self):
        """HTTP publisher should not be available without URL."""
        publisher = HTTPEventPublisher(url="")
        assert publisher.is_available() is False

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_success(self, mock_get_profiler, mock_post):
        """Test successful publish."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        mock_profiler = MagicMock()
        mock_profiler.enabled = False
        mock_get_profiler.return_value = mock_profiler

        publisher = HTTPEventPublisher(url="http://example.com")
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        result = publisher.publish(event)
        assert result is True
        mock_post.assert_called_once()

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_no_url(self, mock_get_profiler, mock_post):
        """Test publish with no URL configured."""
        publisher = HTTPEventPublisher(url="")
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        result = publisher.publish(event)
        assert result is False
        mock_post.assert_not_called()

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_timeout_retry(self, mock_get_profiler, mock_post):
        """Test publish retries on timeout."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        mock_profiler = MagicMock()
        mock_profiler.enabled = False
        mock_get_profiler.return_value = mock_profiler

        publisher = HTTPEventPublisher(
            url="http://example.com",
            retry_count=2,
            retry_delay=0.01,
        )
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        result = publisher.publish(event)
        assert result is False
        assert mock_post.call_count == 2

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_connection_error_retry(self, mock_get_profiler, mock_post):
        """Test publish retries on connection error."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()

        mock_profiler = MagicMock()
        mock_profiler.enabled = False
        mock_get_profiler.return_value = mock_profiler

        publisher = HTTPEventPublisher(
            url="http://example.com",
            retry_count=2,
            retry_delay=0.01,
        )
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        result = publisher.publish(event)
        assert result is False
        assert mock_post.call_count == 2

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_http_error_no_retry_4xx(self, mock_get_profiler, mock_post):
        """Test publish doesn't retry on 4xx errors."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.response = mock_response
        error = requests.exceptions.HTTPError(response=mock_response)
        mock_post.side_effect = error

        mock_profiler = MagicMock()
        mock_profiler.enabled = False
        mock_get_profiler.return_value = mock_profiler

        publisher = HTTPEventPublisher(
            url="http://example.com",
            retry_count=3,
            retry_delay=0.01,
        )
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        result = publisher.publish(event)
        assert result is False
        assert mock_post.call_count == 1  # No retry for 4xx

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_http_error_retry_5xx(self, mock_get_profiler, mock_post):
        """Test publish retries on 5xx errors."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.response = mock_response
        error = requests.exceptions.HTTPError(response=mock_response)
        mock_post.side_effect = error

        mock_profiler = MagicMock()
        mock_profiler.enabled = False
        mock_get_profiler.return_value = mock_profiler

        publisher = HTTPEventPublisher(
            url="http://example.com",
            retry_count=2,
            retry_delay=0.01,
        )
        event = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )

        result = publisher.publish(event)
        assert result is False
        assert mock_post.call_count == 2

    @patch("src.events.http_publisher.requests.post")
    @patch("src.observability.get_profiler")
    def test_publish_batch(self, mock_get_profiler, mock_post):
        """Test batch publish."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        mock_profiler = MagicMock()
        mock_profiler.enabled = False
        mock_get_profiler.return_value = mock_profiler

        publisher = HTTPEventPublisher(url="http://example.com")
        event1 = PlateEvent(
            plate="ABC123",
            plate_normalized="ABC123",
            camera="cam0",
            frames_collected=10,
        )
        event2 = PlateEvent(
            plate="XYZ789",
            plate_normalized="XYZ789",
            camera="cam0",
            frames_collected=5,
        )

        results = publisher.publish_batch([event1, event2])
        assert results == [True, True]
        assert mock_post.call_count == 2


class TestCreateHTTPPublisher:
    """Test publisher factory."""

    def test_create_with_url(self):
        """Should create HTTPEventPublisher with URL."""
        publisher = create_http_publisher(url="http://example.com")
        assert isinstance(publisher, HTTPEventPublisher)
        assert publisher.url == "http://example.com"

    def test_create_without_url(self):
        """Should create NoOpPublisher without URL."""
        publisher = create_http_publisher(url="")
        assert isinstance(publisher, NoOpPublisher)

    def test_create_with_custom_settings(self):
        """Should pass custom settings to publisher."""
        publisher = create_http_publisher(
            url="http://example.com",
            timeout=15.0,
            retry_count=5,
        )
        assert isinstance(publisher, HTTPEventPublisher)
        assert publisher.timeout == 15.0
        assert publisher.retry_count == 5
