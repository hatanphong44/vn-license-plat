# LPR Runtime - 24/7 License Plate Recognition

A modular 24/7 LPR (License Plate Recognition) runtime built on GPU for real-time vehicle plate detection and recognition.

## Architecture

```
Camera/RTSP
    → Frame Capture
    → YOLO Plate Detection
    → Plate Crop
    → OCR Text Detection
    → Text Crop
    → OCR Text Recognition
    → Postprocess (Validation)
    → 3-second Aggregation Window
    → Finalization (majority vote)
    → Deduplication
    → HTTP Event Publisher
    → APP Server
```

## Pipeline Stages

1. **Camera**: Captures frames from USB camera or RTSP stream
2. **Detection**: YOLO plate detector finds vehicle plates
3. **Crop**: Plate regions are cropped for OCR
4. **OCR**: PaddleOCR text detection and recognition
5. **Validation**: Plate text is validated (format, length)
6. **Aggregation**: Observations collected for 3 seconds
7. **Finalization**: Majority vote selects winning plate
8. **Deduplication**: Prevents publishing same plate repeatedly
9. **Publishing**: HTTP callback to application server

## Features

- **Modular Design**: Each model isolated in its own module (per PLAN.md)
- **24/7 Reliability**: Automatic reconnection, error recovery, graceful shutdown
- **GPU Acceleration**: CUDA 13.0 + cuDNN 9 support
- **3-Second Aggregation**: Collects observations for 3 seconds before finalizing
- **Minimum Data Requirement**: Requires at least 10 observations per window
- **Majority Vote**: Most frequent valid plate wins
- **Tie Handling**: Ties result in no result (neither candidate published)
- **Deduplication**: Same plate in consecutive windows is skipped
- **Multi-line Support**: Automatically concatenates multi-line plates with `_`
- **HTTP Events**: Publish plate events to your server
- **Docker Ready**: GPU-enabled container with healthcheck

## Result Aggregation Behavior

### Window Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RESULT_WINDOW_SECONDS` | 3.0 | Duration of each aggregation window |
| `MIN_OBSERVATIONS_PER_WINDOW` | 10 | Minimum observations required for finalization |

### Aggregation Rules

```
Window Duration = 3 seconds (time-based, not frame-count based)
```

```
Collect observations for 3 seconds:
    |
    |--- fewer than 10 observations
    |     → NO FINAL RESULT
    |     → NO PUBLISH
    |
    |--- 10 or more observations
          → Apply majority vote
          → Check for ties
          → Publish if winner
```

### Finalization Logic

For each 3-second window with ≥10 observations:

1. **Validate observations**: Filter out invalid plate formats
2. **Count plate occurrences**: Count only valid observations
3. **Select winner**:
   - Most frequent valid plate wins
   - If tie between top 2 candidates → NO RESULT
4. **Deduplicate**: Skip if same as last published plate
5. **Publish**: Send to callback URL if new plate

### Examples

```
3 seconds + 5 observations → NO RESULT (insufficient data)
3 seconds + 9 observations → NO RESULT (insufficient data)
3 seconds + 10 observations → FINALIZATION ALLOWED
3 seconds + 15 observations → FINALIZATION ALLOWED
```

**Note**: Confidence does NOT bypass the minimum 10 observation requirement.

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

### 4. Test on Camera

```bash
# Test on camera (default device 0)
python scripts/test_models.py --camera

# Test on specific camera device
python scripts/test_models.py --camera --camera-id 1

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
│   │   ├── worker.py         # Main worker loop (3-second windows)
│   │   └── controller.py     # Runtime controller
│   ├── api/                  # FastAPI server
│   │   └── server.py         # API endpoints
│   ├── camera/               # Camera abstraction
│   │   ├── base.py           # Base camera interface
│   │   ├── usb.py            # USB camera
│   │   └── rtsp.py           # RTSP camera
│   ├── visualization/        # Overlay rendering
│   │   ├── annotator.py      # Frame annotation
│   │   ├── overlay.py        # Real-time overlay
│   │   └── state.py          # Visualization state
│   ├── domain/               # Domain models
│   │   └── models.py         # Typed objects (LPRResult, etc.)
│   ├── config/               # Settings
│   │   └── settings.py       # Pydantic settings
│   └── logging/              # Logging setup
├── scripts/
│   └── test_models.py       # Camera/image model testing
├── tests/                    # Unit/integration tests
├── models/                   # Model weights (gitignored)
├── configs/                  # Configuration files
├── Dockerfile                # Multi-stage GPU container
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

## Result Selection Algorithm

Per PLAN.md specification:

1. **3-second window expires** → finalize window
2. **Check minimum observations** → require ≥10 observations
3. **Filter valid observations** → exclude invalid plate formats
4. **Count occurrences** → count valid plate texts
5. **Select winner**:
   - Most frequent valid plate wins
   - If tie between top 2 candidates → NO RESULT
6. **Deduplicate** → skip if same as last published
7. **Publish** → send to callback URL

```
Scoring algorithm:
1. Count occurrences of each plate text (valid only)
2. Pick the one with most votes (majority vote)
3. If tie, return NO RESULT (no winner)
4. If still tie, pick longest text
```

## License Plate Format

- **Multi-line plates**: concatenated with `_` separator
  - Example: `"29A"` + `"12345"` → `"29A_12345"`
- **Normalized**: uppercase, letters/numbers only
- **Validation**: 4-15 characters after removing `_`

## Visualization

The project includes real-time visualization capabilities:

### Overlay Rendering (`src/visualization/`)

- **Real-time display**: Shows camera feed with detection overlays
- **Non-blocking**: Runs in separate thread
- **Headless mode**: Saves frames to disk when no display available

### Annotator (`src/visualization/annotator.py`)

- `ResultAnnotator`: Draws plate boxes and text on frames
- `PlateDetectionAnnotator`: Draws intermediate detection stages

### Frame Annotation

```
┌─────────────────────────────────┐
│ FPS: 10.5                       │
│                                 │
│  ┌──────────┐                   │
│  │ 92CA03484│ (0.95)            │
│  └──────────┘                   │
│                                 │
└─────────────────────────────────┘
```

### Debugging with Visualization

```bash
# Enable visualization
export VISUALIZE=1

# Enable headless mode (save to disk)
python -m src.main --headless

# Custom save directory
export CAPTURE_DIR=./debug_captures
```

### Headless Mode

When no display is available (WSL, SSH, server), the renderer saves annotated frames to disk:

```
captures/
├── capture_20240101_120000_plates_1.jpg
├── capture_20240101_120005_no_plate.jpg
└── ...
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_window.py -v
```

### Key Test Files

| Test File | Description |
|-----------|-------------|
| `tests/unit/test_window.py` | Window aggregation and finalization logic |
| `tests/unit/test_pipeline.py` | LPR pipeline processing |
| `tests/unit/test_camera.py` | Camera abstraction |
| `tests/unit/test_runtime_controller.py` | Runtime controller |

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

- Check device number: `ls /dev/video*`
- Try different device: `export CAMERA_SOURCE=1`
- Verify RTSP URL format: `rtsp://ip:port/stream`
- Check network connectivity for RTSP cameras
- Check camera credentials if required

### Model loading errors

```bash
# Verify model files exist
ls -la models/

# Check model directory structure
ls -la models/PP-OCRv6_small_det/
ls -la models/PP-OCRv6_small_rec/
```

### OCR failure / No text detected

- Verify plate is clearly visible in frame
- Check lighting conditions
- Verify text padding: `TEXT_PADDING=10`
- Verify upscale factor: `OCR_UPSCALE=4`

### Validation rejection / No plate results

- Check plate format matches expected pattern (4-15 alphanumeric characters)
- Multi-line plates should be concatenated with `_`
- Enable debug logging: `export LOG_LEVEL=DEBUG`

### Insufficient observations / No result published

- **Minimum 10 observations required** per 3-second window
- Check camera FPS: `INFERENCE_FPS` should be realistic
- Verify plate is in frame for at least 3 seconds
- Enable debug logging to see observation counts

### Publisher/API failures

- Verify `CALLBACK_URL` is accessible
- Check network connectivity to callback server
- Verify callback endpoint accepts POST requests
- Check callback timeout: `CALLBACK_TIMEOUT=5.0`

### Permission problems

```bash
# Camera access (Linux)
sudo chmod 666 /dev/video0

# GPU access
# Ensure user is in video group
groups $USER
sudo usermod -a -G video $USER
```

### WSL / Headless environment

```bash
# Use headless mode
python -m src.main --headless

# Or disable visualization
export VISUALIZE=0
```

### Debugging the Pipeline

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python -m src.main 2>&1 | grep -E "(OBS|Window|PUBLISH|plate)"
```

Example debug output:

```
[DEBUG] Window 1: 5 observations (3 valid)
[DEBUG] Window 1: insufficient observations (5/10)
[DEBUG] Window 2: 12 observations (10 valid)
[INFO] [EVENT] Published: plate=92CA03484 conf=0.923
```

## License

MIT
