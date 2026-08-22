"""Tests for src/api/server.py - full coverage"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient


class TestAPIServer:
    """Test API server endpoints."""

    def test_create_app(self):
        """Test create_app function."""
        from src.api.server import create_app

        app = create_app()
        assert app is not None

    def test_health_endpoint(self):
        """Test /health endpoint."""
        from src.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_ready_endpoint_without_pipeline(self):
        """Test /ready endpoint without pipeline returns 503."""
        from src.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert "status" in data["detail"]
        assert "ready" in data["detail"]
        assert data["detail"]["ready"] is False

    def test_ready_endpoint_with_pipeline(self):
        """Test /ready endpoint with pipeline returns 200."""
        from src.api.server import create_app

        # Create mock pipeline with all components
        mock_pipeline = MagicMock()
        mock_pipeline.plate_detector = MagicMock()
        mock_pipeline.text_detector = MagicMock()
        mock_pipeline.text_recognizer = MagicMock()

        app = create_app(pipeline=mock_pipeline)
        client = TestClient(app)

        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "gpu" in data

    def test_root_endpoint(self):
        """Test / endpoint."""
        from src.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LPR Runtime API"

    def test_metrics_endpoint(self):
        """Test /metrics endpoint."""
        from src.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "runtime_stats" in data


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_no_pipeline(self):
        """Test /predict without pipeline returns 503."""
        from src.api.server import create_app

        app = create_app(pipeline=None)
        client = TestClient(app)

        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode('.jpg', img)
        files = {"file": ("test.jpg", BytesIO(img_bytes), "image/jpeg")}

        response = client.post("/predict", files=files)
        assert response.status_code == 503

    def test_predict_invalid_content_type(self):
        """Test /predict with invalid content type."""
        from src.api.server import create_app

        app = create_app(pipeline=MagicMock())
        client = TestClient(app)

        # Send non-image file
        files = {"file": ("test.txt", BytesIO(b"not image data"), "text/plain")}

        response = client.post("/predict", files=files)
        assert response.status_code == 415

    def test_predict_invalid_image_data(self):
        """Test /predict with invalid image data."""
        from src.api.server import create_app

        app = create_app(pipeline=MagicMock())
        client = TestClient(app)

        # Create corrupt image data
        files = {"file": ("test.jpg", BytesIO(b"not a valid jpeg"), "image/jpeg")}

        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_predict_success(self):
        """Test /predict with valid image."""
        from src.api.server import create_app

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.plate = "ABC123"
        mock_result.plate_normalized = "ABC123"
        mock_result.box = [10, 10, 100, 50]
        mock_result.yolo_score = 0.95
        mock_result.get_confidence.return_value = 0.90
        mock_result.ocr_results = []
        mock_pipeline.process_frame.return_value = [mock_result]

        app = create_app(pipeline=mock_pipeline)
        client = TestClient(app)

        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode('.jpg', img)
        files = {"file": ("test.jpg", BytesIO(img_bytes.tobytes()), "image/jpeg")}

        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "plates" in data
        assert data["count"] == 1


class TestRuntimeEndpoints:
    """Test runtime control endpoints."""

    def test_stop_runtime_not_running(self):
        """Test /stop-runtime when not running."""
        from src.api.server import create_app
        from src.runtime.controller import RuntimeController

        controller = RuntimeController()
        app = create_app(controller=controller)
        client = TestClient(app)

        response = client.post("/stop-runtime")
        data = response.json()
        assert data["status"] == "not_running"

    def test_stop_alias(self):
        """Test /stop alias for /stop-runtime."""
        from src.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post("/stop")
        assert response.status_code == 200


class TestStartRuntimeFull:
    """Test start-runtime endpoint full coverage."""

    @patch("src.camera.create_camera")
    @patch("src.events.create_http_publisher")
    @patch("src.config.get_settings")
    def test_start_runtime_camera_connect_fails(
        self, mock_get_settings, mock_create_publisher, mock_create_camera
    ):
        """Test /start-runtime when camera fails to connect."""
        from src.api.server import create_app

        mock_settings = MagicMock()
        mock_settings.camera.CAMERA_SOURCE = "0"
        mock_settings.camera.CAMERA_BUFFER_SIZE = 1
        mock_settings.camera.CAMERA_CONNECT_TIMEOUT = 10
        mock_settings.camera.CAMERA_RECONNECT_DELAY = 3.0
        mock_settings.events.CALLBACK_URL = ""
        mock_settings.events.CALLBACK_TIMEOUT = 5.0
        mock_settings.events.CALLBACK_RETRY_COUNT = 3
        mock_settings.events.CALLBACK_RETRY_DELAY = 1.0

        mock_get_settings.return_value = mock_settings

        mock_camera = MagicMock()
        mock_camera.connect.return_value = False
        mock_camera.disconnect.return_value = None
        mock_create_camera.return_value = mock_camera

        mock_controller = MagicMock()
        mock_controller.is_running = False

        app = create_app(controller=mock_controller)
        client = TestClient(app)

        response = client.post("/start-runtime")
        assert response.status_code == 503
        assert "Cannot connect to camera" in response.json()["detail"]

    @patch("src.camera.create_camera")
    @patch("src.events.create_http_publisher")
    @patch("src.config.get_settings")
    def test_start_runtime_success(
        self, mock_get_settings, mock_create_publisher, mock_create_camera
    ):
        """Test /start-runtime successfully starts runtime."""
        from src.api.server import create_app

        mock_settings = MagicMock()
        mock_settings.camera.CAMERA_SOURCE = "0"
        mock_settings.camera.CAMERA_BUFFER_SIZE = 1
        mock_settings.camera.CAMERA_CONNECT_TIMEOUT = 10
        mock_settings.camera.CAMERA_RECONNECT_DELAY = 3.0
        mock_settings.events.CALLBACK_URL = "http://example.com/callback"
        mock_settings.events.CALLBACK_TIMEOUT = 5.0
        mock_settings.events.CALLBACK_RETRY_COUNT = 3
        mock_settings.events.CALLBACK_RETRY_DELAY = 1.0

        mock_get_settings.return_value = mock_settings

        mock_camera = MagicMock()
        mock_camera.connect.return_value = True
        mock_camera.source = "0"
        mock_create_camera.return_value = mock_camera

        mock_publisher = MagicMock()
        mock_create_publisher.return_value = mock_publisher

        mock_controller = MagicMock()
        mock_controller.is_running = False

        mock_pipeline = MagicMock()
        app = create_app(controller=mock_controller, pipeline=mock_pipeline)
        client = TestClient(app)

        response = client.post("/start-runtime")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["camera_source"] == "0"

    @patch("src.camera.create_camera")
    @patch("src.events.create_http_publisher")
    @patch("src.config.get_settings")
    def test_start_runtime_exception_handling(
        self, mock_get_settings, mock_create_publisher, mock_create_camera
    ):
        """Test /start-runtime handles unexpected exceptions."""
        from src.api.server import create_app

        mock_settings = MagicMock()
        mock_settings.camera.CAMERA_SOURCE = "0"
        mock_settings.camera.CAMERA_BUFFER_SIZE = 1
        mock_settings.camera.CAMERA_CONNECT_TIMEOUT = 10
        mock_settings.camera.CAMERA_RECONNECT_DELAY = 3.0
        mock_settings.events.CALLBACK_URL = ""
        mock_settings.events.CALLBACK_TIMEOUT = 5.0
        mock_settings.events.CALLBACK_RETRY_COUNT = 3
        mock_settings.events.CALLBACK_RETRY_DELAY = 1.0

        mock_get_settings.return_value = mock_settings

        mock_camera = MagicMock()
        mock_camera.connect.side_effect = RuntimeError("Camera error")
        mock_create_camera.return_value = mock_camera

        mock_controller = MagicMock()
        mock_controller.is_running = False

        app = create_app(controller=mock_controller)
        client = TestClient(app)

        response = client.post("/start-runtime")
        assert response.status_code == 500
        assert "Failed to start runtime" in response.json()["detail"]


class TestStopRuntimeFull:
    """Test stop-runtime endpoint full coverage."""

    def test_stop_runtime_exception_handling(self):
        """Test /stop-runtime handles exceptions gracefully."""
        from src.api.server import create_app

        mock_controller = MagicMock()
        mock_controller.is_running = True
        mock_controller.stop.side_effect = RuntimeError("Stop error")

        mock_camera = MagicMock()
        mock_camera.disconnect.return_value = None

        app = create_app(controller=mock_controller)
        app.state.camera = mock_camera
        client = TestClient(app)

        response = client.post("/stop-runtime")
        assert response.status_code == 500
        assert "Failed to stop runtime" in response.json()["detail"]

    @patch("src.api.server.get_controller")
    def test_stop_runtime_with_camera_cleanup(self, mock_get_controller):
        """Test /stop-runtime properly cleans up camera."""
        from src.api.server import create_app

        mock_controller = MagicMock()
        mock_controller.is_running = True
        mock_controller.stop.return_value = None

        mock_get_controller.return_value = mock_controller

        mock_camera = MagicMock()
        mock_camera.disconnect.return_value = None
        mock_camera.source = "0"

        app = create_app()
        app.state.camera = mock_camera
        client = TestClient(app)

        response = client.post("/stop-runtime")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert app.state.camera is None
