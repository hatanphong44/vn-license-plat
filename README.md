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

- **Modular Design**: Each model isolated in its own module (per PLAN.md)
- **24/7 Reliability**: Automatic reconnection, error recovery, graceful shutdown
- **GPU Acceleration**: CUDA 13.0 + cuDNN 9 support
- **Best Result Selection**: Collect up to 20 frames per plate, pick the best
- **Multi-line Support**: Automatically concatenates multi-line plates with `_`
- **HTTP Events**: Publish plate events to your server
- **Docker Ready**: GPU-enabled container with healthcheck

## Models

| Model | Type | Description |
|-------|------|-------------|
| `Plate.pt` | YOLO | Plate detector (confidence, IoU threshold) |
| `PP-OCRv6_small_det` | PaddleOCR | Text region detector |
| `PP-OCRv6_small_rec` | PaddleOCR | Text recognizer |

## Quick Start

### 1. Environment Setup (GPU with CUDA 13.0)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA 13.0
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install PaddlePaddle GPU
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# Install other dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
```

### 2. Download Models

Place model files in `models/` directory:
```
models/
├── Plate.pt
├── PP-OCRv6_small_det/
│   ├── inference.pdiparams
│   ├── inference.yml
│   └── inference.json
└── PP-OCRv6_small_rec/
    ├── inference.pdiparams
    ├── inference.yml
    └── inference.json
```

### 3. Test on Single Image

```bash
python -m src.test_image --image path/to/image.jpg
```

With options:
```bash
python -m src.test_image \
    --image path/to/image.jpg \
    --yolo-model ./models/Plate.pt \
    --paddle-model-dir ./models \
    --yolo-device 0 \
    --paddle-device gpu:0 \
    --visualize
```

### 4. Test on Video

```bash
# Test on video file
python scripts/test_models.py --video path/to/video.mp4 --output result.mp4

# Test on camera
python scripts/test_models.py --camera

# Test single image
python scripts/test_models.py --image path/to/image.jpg
```

### 5. Run Runtime

```bash
# Set environment variables
export CAMERA_SOURCE=0
export YOLO_MODEL_PATH=./models/Plate.pt
export PADDLE_MODEL_DIR=./models
export CALLBACK_URL=http://your-server.com/api/plates
export INFERENCE_FPS=5.0

# Run 24/7 runtime
python -m src.main
```

### 6. Run API Server

```bash
# Start API server
python -m src.main --mode api

# Or with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 7. Docker

```bash
# Build
docker build -t lpr-runtime .

# Run
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/models \
    -e CAMERA_SOURCE=0 \
    lpr-runtime

# Or with docker-compose
docker-compose up -d
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `./models/Plate.pt` | YOLO model path |
| `YOLO_DEVICE` | `0` | YOLO GPU device (0, 1, cpu) |
| `YOLO_CONF` | `0.25` | YOLO confidence threshold |
| `YOLO_IOU` | `0.45` | YOLO IoU threshold |
| `PADDLE_MODEL_DIR` | `./models` | PaddleOCR models directory |
| `PADDLE_DEVICE` | `gpu:0` | PaddleOCR device |
| `OCR_UPSCALE` | `4` | Plate upscale factor for OCR |
| `TEXT_PADDING` | `10` | Padding around text regions |
| `REC_MIN_SCORE` | `0.0` | Recognition minimum score |
| `CAMERA_SOURCE` | `0` | Camera/RTSP URL |
| `INFERENCE_FPS` | `5.0` | Target inference FPS |
| `CALLBACK_URL` | - | HTTP callback URL for events |
| `CALLBACK_TIMEOUT` | `5.0` | HTTP callback timeout |
| `PLATE_COOLDOWN_SECONDS` | `30.0` | Cooldown after sending |
| `MAX_CAPTURE_FRAMES` | `20` | Frames to collect per plate |
| `MAX_CAPTURE_WAIT_SECONDS` | `10.0` | Max wait before sending |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Runtime metrics |
| `/predict` | POST | Manual image test (multipart/form-data) |
| `/stop` | POST | Stop runtime |
| `/start` | POST | Check/start status |

## Project Structure

```
lpr-runtime/
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point (runtime/api modes)
│   ├── test_image.py        # Single image testing
│   ├── models/              # ML model adapters (isolated)
│   │   ├── plate_detector/  # YOLO plate detector
│   │   ├── text_detector/   # PaddleOCR text detector
│   │   └── text_recognizer/ # PaddleOCR text recognizer
│   ├── pipeline/            # LPR inference pipeline
│   │   ├── lpr_pipeline.py  # Main pipeline
│   │   ├── cropper.py       # Crop/preprocessing
│   │   └── postprocessor.py  # Text normalization
│   ├── events/              # Event publishing
│   │   ├── plate_collector.py  # Collect frames, pick best
│   │   ├── publisher.py      # Base publisher
│   │   └── http_publisher.py # HTTP event publisher
│   ├── runtime/              # 24/7 worker
│   │   ├── worker.py         # Main worker loop
│   │   └── controller.py     # Runtime controller
│   ├── api/                  # FastAPI server
│   │   └── server.py         # API endpoints
│   ├── camera/               # Camera abstraction
│   │   ├── base.py           # Base camera interface
│   │   ├── usb.py            # USB camera
│   │   ├── rtsp.py           # RTSP camera
│   │   └── video.py          # Video file
│   ├── visualization/        # Overlay rendering
│   ├── domain/               # Domain models
│   │   └── models.py         # Typed objects (LPRResult, etc.)
│   ├── config/               # Settings
│   │   └── settings.py       # Pydantic settings
│   └── logging/              # Logging setup
├── scripts/
│   └── test_models.py       # Video/image model testing
├── tests/                    # Unit/integration tests
├── models/                   # Model weights (gitignored)
├── configs/                  # Configuration files
├── Dockerfile                # Multi-stage GPU container
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

## Best Result Selection

Per PLAN.md specification:

1. **New plate detected** → start collecting frames
2. **Same plate detected** → add to collection (up to 20 frames or 10 seconds)
3. **Collection complete** → score all results → pick best
4. **Send to server** → cooldown period (30 seconds default)

Scoring algorithm:
```
1. Count occurrences of each plate text
2. Pick the one with most votes (majority vote)
3. If tie, break by confidence score
4. If still tie, pick longest text
```

## License Plate Format

- **Multi-line plates**: concatenated with `_` separator
  - Example: `"29A"` + `"12345"` → `"29A_12345"`
- **Normalized**: uppercase, letters/numbers only
- **Validation**: 4-15 characters after removing `_`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_postprocessor.py -v
```

## Troubleshooting

### GPU not detected
```bash
# Check CUDA
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verify PaddlePaddle CUDA
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
```

### Camera connection failed
- Check RTSP URL format: `rtsp://ip:port/stream`
- Verify network connectivity
- Check camera credentials

### Model loading errors
```bash
# Verify model files exist
ls -la models/

# Check model directory structure
ls -la models/PP-OCRv6_small_det/
ls -la models/PP-OCRv6_small_rec/
```

## License

MIT
