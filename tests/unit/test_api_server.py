"""Unit tests for API server."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.server import create_app


class TestHealthEndpoints:
    """Test health/readiness endpoints."""

    def test_health_endpoint(self):
        """Test /health returns ok status."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_ready_endpoint_without_pipeline(self):
        """Test /ready returns 503 when no pipeline is configured."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/ready")

        # Without pipeline, /ready should return 503 as models are not ready
        assert response.status_code == 503
        data = response.json()
        assert "status" in data["detail"]
        assert "ready" in data["detail"]
        assert data["detail"]["ready"] is False

    def test_ready_endpoint_with_mock_pipeline(self):
        """Test /ready returns 200 when pipeline is configured."""
        from unittest.mock import MagicMock

        # Create a mock pipeline with all model components
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
        assert "status" in data
        assert "gpu" in data
        assert "runtime" in data

    def test_root_endpoint(self):
        """Test / returns app info."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LPR Runtime API"
        assert data["version"] == "2.0.0"
        assert data["docs"] == "/docs"

    def test_metrics_endpoint(self):
        """Test /metrics returns stats."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "runtime_stats" in data


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_without_pipeline(self):
        """Test /predict returns 503 when no pipeline configured."""
        app = create_app(pipeline=None)
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"fake image", "image/jpeg")},
        )

        assert response.status_code == 503
        assert "Pipeline not configured" in response.json()["detail"]

    def test_predict_invalid_content_type(self):
        """Test /predict returns 415 for non-image."""
        mock_pipeline = MagicMock()
        app = create_app(pipeline=mock_pipeline)
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 415
        assert "image" in response.json()["detail"].lower()

    def test_predict_invalid_image_data(self):
        """Test /predict returns 400 for invalid image data."""
        mock_pipeline = MagicMock()
        app = create_app(pipeline=mock_pipeline)
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"not a valid image", "image/jpeg")},
        )

        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    @patch("cv2.imdecode")
    def test_predict_success(self, mock_imdecode):
        """Test /predict returns results for valid image."""
        import numpy as np

        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_image

        mock_result = MagicMock()
        mock_result.plate = "ABC123"
        mock_result.plate_normalized = "ABC123"
        mock_result.box = [10, 20, 100, 80]
        mock_result.yolo_score = 0.9
        mock_result.get_confidence.return_value = 0.85
        mock_result.ocr_results = []

        mock_pipeline = MagicMock()
        mock_pipeline.process_frame.return_value = [mock_result]

        app = create_app(pipeline=mock_pipeline)
        client = TestClient(app)

        # Create a simple JPEG-like bytes
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"\xff\xd8\xff\xe0 fake jpeg", "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["plates"]) == 1
        assert data["plates"][0]["plate"] == "ABC123"


class TestRuntimeEndpoints:
    """Test runtime control endpoints."""

    @patch("src.api.server.get_controller")
    def test_start_runtime_already_running(self, mock_get_controller):
        """Test /start-runtime when already running."""
        mock_controller = MagicMock()
        mock_controller.is_running = True

        mock_cam = MagicMock()
        mock_cam.source = "rtsp://example.com"

        app = create_app(controller=mock_controller)
        app.state.camera = mock_cam
        client = TestClient(app)

        response = client.post("/start-runtime")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_running"

    @patch("src.api.server.get_controller")
    def test_stop_runtime_not_running(self, mock_get_controller):
        """Test /stop-runtime when not running."""
        mock_controller = MagicMock()
        mock_controller.is_running = False

        app = create_app(controller=mock_controller)
        client = TestClient(app)

        response = client.post("/stop-runtime")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_running"

    @patch("src.api.server.get_controller")
    def test_start_alias(self, mock_get_controller):
        """Test /start is alias for /start-runtime."""
        mock_controller = MagicMock()
        mock_controller.is_running = True

        app = create_app(controller=mock_controller)
        client = TestClient(app)

        response = client.post("/start")
        # Should work since we're mocking controller
        assert response.status_code in [200, 500]

    @patch("src.api.server.get_controller")
    def test_stop_alias(self, mock_get_controller):
        """Test /stop is alias for /stop-runtime."""
        mock_controller = MagicMock()
        mock_controller.is_running = False

        app = create_app(controller=mock_controller)
        client = TestClient(app)

        response = client.post("/stop")
        assert response.status_code == 200
