# LPR Runtime - 24/7 License Plate Recognition

A modular 24/7 LPR (License Plate Recognition) runtime built on GPU for real-time vehicle plate detection and recognition.

## Architecture

```
Camera/RTSP → Frame Capture → YOLO Plate Detection
→ Plate Crop → OCR Text Detection → Text Crop
→ OCR Text Recognition → Postprocess
→ New Plate/Best-Result Selection → HTTP Event Publisher → APP Server
```

## Features

- **Modular Design**: Each model isolated in its own module
- **24/7 Reliability**: Automatic reconnection, error recovery, graceful shutdown
- **GPU Acceleration**: CUDA support for all models
- **Best Result Selection**: Collect up to 20 frames per plate, pick the best
- **Multi-line Support**: Automatically concatenates multi-line plates
- **HTTP Events**: Publish plate events to your server
- **Docker Ready**: GPU-enabled container with healthcheck

## Models

- **Plate.pt** - YOLO plate detector
- **PP-OCRv6_small_det** - PaddleOCR text detector
- **PP-OCRv6_small_rec** - PaddleOCR text recognizer

## Quick Start

### 1. Environment Setup (GPU)

```bash
# Install dependencies
pip install -U paddleocr
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
pip install -q -U ultralytics

# Verify GPU
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
```

### 2. Test Models on Video

```bash
# Test on video file
python scripts/test_models.py --video path/to/video.mp4 --output result.mp4

# Test on camera
python scripts/test_models.py --camera 0

# Verify GPU setup
python scripts/test_models.py --verify
```

### 3. Run Runtime

```bash
# Set environment variables
export CAMERA_SOURCE=rtsp://192.168.1.100:554/stream
export CALLBACK_URL=http://your-server.com/api/plates

# Run
python -m src.main
```

### 4. Run API Server

```bash
# Start API server
python -m src.main --mode api

# Or with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 5. Docker

```bash
# Build
docker build -t lpr-runtime .

# Run with docker-compose
docker-compose up -d

# With environment file
cp .env.example .env
# Edit .env with your settings
docker-compose up -d
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `/models/plate_detector/Plate.pt` | YOLO model path |
| `YOLO_DEVICE` | `0` | GPU device |
| `PADDLE_DEVICE` | `gpu:0` | PaddleOCR device |
| `CAMERA_SOURCE` | `0` | Camera/RTSP URL |
| `INFERENCE_FPS` | `5.0` | Target FPS |
| `CALLBACK_URL` | - | HTTP callback URL |
| `PLATE_COOLDOWN_SECONDS` | `30.0` | Cooldown after sending |
| `MAX_CAPTURE_FRAMES` | `20` | Frames to collect per plate |
| `LOG_LEVEL` | `INFO` | Log level |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Runtime metrics |
| `/predict` | POST | Manual image test |
| `/stop` | POST | Stop runtime |
| `/start` | POST | Check status |

## Project Structure

```
lpr-runtime/
├── src/
│   ├── models/           # ML model adapters
│   │   ├── plate_detector/
│   │   ├── text_detector/
│   │   └── text_recognizer/
│   ├── camera/           # Camera abstraction
│   ├── pipeline/         # LPR inference pipeline
│   ├── events/           # Event publishing
│   ├── runtime/          # Worker and controller
│   ├── api/              # FastAPI server
│   ├── visualization/    # Overlay rendering
│   ├── domain/           # Domain models
│   ├── config/           # Settings
│   └── logging/          # Logging setup
├── scripts/              # Utilities
├── tests/                # Unit/integration tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Best Result Selection

1. New plate detected → start collecting frames
2. Same plate detected → add to collection (up to 20 frames)
3. Collection complete → score all results → pick best
4. Send to server → cooldown period

Scoring:
- Majority vote (most common text wins)
- Confidence score tiebreaker
- Longest text tiebreaker

## License Plate Format

- Multi-line plates: concatenated with `_` (e.g., `29A_12345`)
- Normalized: uppercase, letters/numbers only

## Monitoring

```bash
# Check health
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

## Troubleshooting

### GPU not detected
```bash
python scripts/test_models.py --verify
```

### Camera connection failed
- Check RTSP URL format: `rtsp://ip:port/stream`
- Verify network connectivity
- Check camera credentials

### Model loading errors
- Verify model files exist at configured paths
- Check model file permissions
- Ensure correct model format (YOLO .pt, PaddleOCR models)

## License

MIT
