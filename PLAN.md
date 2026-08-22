# LPR Runtime — PLAN

## Goal
Build a modular 24/7 LPR inference runtime on a GPU machine.

```text
Camera/RTSP → Frame Capture → YOLO Plate Detection
→ Plate Crop → OCR Text Detection → Text Crop
→ OCR Text Recognition → Postprocess
→ New Plate/Best-Result Selection → HTTP Event Publisher → APP Server
```

## Models
- `Plate.pt` — YOLO plate detector
- `PP-OCRv6_small_det` — OCR text detector
- `PP-OCRv6_small_rec` — OCR text recognizer

Each model must be isolated in its own module.

### Text Concatenation (Multi-line Plates)
OCR text detector có thể detect ra nhiều dòng text trên cùng 1 biển số. Xử lý:
- Nếu detector trả về **2 dòng** → nối bằng `_` (ví dụ: `29A` + `12345` → `29A_12345`)
- Nếu detector trả về **1 dòng** → giữ nguyên
- Thứ tự ghép: từ trên xuống dưới (top → bottom)

```python
# Example output from text_detector
detections = [
    {"text": "29A", "bbox": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], "confidence": 0.95},  # top line
    {"text": "12345", "bbox": [[x1,y2], [x2,y2], [x2,y3], [x1,y3]], "confidence": 0.92},  # bottom line
]
# Concatenate: "29A_12345"
```

## Folder Structure

```text
lpr-runtime/
├── src/
│   ├── models/
│   │   ├── plate_detector/
│   │   ├── text_detector/
│   │   └── text_recognizer/
│   ├── camera/
│   │   ├── base.py
│   │   ├── usb.py
│   │   ├── rtsp.py
│   │   └── video.py
│   ├── pipeline/
│   │   ├── lpr_pipeline.py
│   │   ├── cropper.py
│   │   └── postprocessor.py
│   ├── events/
│   │   ├── publisher.py
│   │   ├── http_publisher.py
│   │   └── plate_collector.py
│   ├── runtime/
│   │   ├── worker.py
│   │   └── controller.py
│   ├── api/
│   │   └── server.py
│   ├── visualization/
│   │   ├── annotator.py      # Draw boxes, text, FPS
│   │   ├── overlay.py        # Real-time camera overlay
│   │   └── state.py
│   ├── domain/
│   │   └── models.py
│   ├── config/
│   │   └── settings.py
│   ├── logging/
│   │   └── setup.py
│   └── observability/
│       └── metrics.py
├── configs/
├── models/
│   ├── Plate.pt                 # YOLO plate detector
│   ├── PP-OCRv6_small_det/     # OCR text detector
│   └── PP-OCRv6_small_rec/     # OCR text recognizer
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── scripts/
│   └── test_models.py       # Video-based model testing tool
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```



## Responsibilities

**Models:** load/inference only. No camera, HTTP, FastAPI, Docker, or DB logic.

**Camera:** connect, read, reconnect, release, health. Pipeline must not directly manage `cv2.VideoCapture`.

**Pipeline:** `Frame → Detection → Crop → OCR → Postprocess → LPRResult`. No HTTP.

**Events:** event creation, publishing. Publisher must be replaceable by HTTP/Redis/Kafka later.

**Runtime Worker:** 24/7 loop, lifecycle, recovery, reconnect, graceful shutdown.

**API:** `/health`, `/ready`, `/metrics`, optional `/predict`. No camera loop.

**Visualization:** Overlay rendering, must not block inference loop. Run in separate thread.

**Logging:** Structured console output with timestamps. No external dependencies (no Logstash, no ELK).

## Domain Objects
Use typed models:
- `PlateDetection`
- `TextDetection`
- `TextRecognition`
- `LPRResult`
- `CapturedPlate` (single plate detection with frame image)
- `PlateCollection` (group of 20 frames for same plate)
- `PlateEvent`

## Logging

Terminal output format: `[LEVEL] message` with timestamps.

### Startup Logs

```
{HH:MM:SS} [INFO] LPR Runtime starting
{HH:MM:SS} [INFO] CUDA device: NVIDIA RTX ...
{HH:MM:SS} [INFO] Plate model loaded
{HH:MM:SS} [INFO] OCR detector loaded
{HH:MM:SS} [INFO] OCR recognizer loaded
{HH:MM:SS} [INFO] Camera connected
{HH:MM:SS} [INFO] Runtime ready
```

### Runtime Logs

```
{HH:MM:SS} [INFO] New plate detected: {plate_number}
{HH:MM:SS} [INFO] Plate collection completed: plate={plate_number} frames={count}
{HH:MM:SS} [INFO] Best result: plate={plate_number} confidence={score}
{HH:MM:SS} [INFO] Publishing event: plate={plate_number}
{HH:MM:SS} [INFO] Event published: plate={plate_number} status={http_code}
```

### Error Logs

```
{HH:MM:SS} [ERROR] Camera disconnected
{HH:MM:SS} [ERROR] Inference error: {details}
{HH:MM:SS} [ERROR] Publish failed: plate={plate_number} error={details}
```

### Debug Logs (optional, via DEBUG flag)

```
{HH:MM:SS} [DEBUG] Frame captured: width={w} height={h}
{HH:MM:SS} [DEBUG] Detection found: plate={plate_number} confidence={score}
{HH:MM:SS} [DEBUG] Collection progress: plate={plate_number} frames={count}/20
```

## Best-Result Selection
When a **new plate number** is detected:

1. Capture the first detection → start collecting frames
2. Continue capturing up to `MAX_CAPTURE_FRAMES` (default: 20) detections of the **same plate**
3. After collection completes, run OCR on all captured frames
4. Score each OCR result and return **the best one** to the server

Example flow:

```text
Frame 1:  → 29A12345 → collect (1/20)
Frame 2:  → 29A12345 → collect (2/20)
Frame 3:  → 30B67890 → NEW plate! → start new collection
Frame 4:  → 29A12345 → collect (3/20) for 29A12345
...
Frame 20: → 29A12345 → collect (20/20) → run OCR on all 20, pick best → send to server
Frame 21: → 29A12345 → COOLDOWN (already sent)
```

Scoring criteria (configurable):
- **Majority vote** — kết quả xuất hiện nhiều nhất được chọn (prioritized)
- Highest confidence score from OCR model
- Text format validation (length, character set)
- Lexicon matching (if enabled)

**Algorithm:**
```python
# 1. Count occurrences of each plate text
text_counts = Counter(detected_plates)  # e.g., {"29A12345": 15, "29A1234": 3, "29A123": 2}

# 2. If unique plate numbers exist, pick the one with most votes
# 3. If tie, break by confidence score
# 4. If still tie, pick longest text (more characters = more likely correct)
```

Configurable parameters:
- `PLATE_COOLDOWN_SECONDS` — cooldown after sending
- `MAX_CAPTURE_FRAMES` — how many frames to collect per plate
- `MAX_CAPTURE_WAIT_SECONDS` — max time to wait before sending (in case camera moves away)

Interface design:

```python
class PlateCollector:
    def add_detection(self, plate_text: str, confidence: float) -> None
    def is_complete(self) -> bool
    def get_best_result(self) -> CapturedPlate  # Returns plate with most votes
    def get_all_results(self) -> dict[str, list[CapturedPlate]]  # All results grouped by text
    def should_start_new_collection(self) -> bool
    def clear(self) -> None
```

## Visualization

### Real-time Overlay
Draw inference results directly on the camera feed for live monitoring:
- Plate detection bounding boxes
- OCR text overlay on plates
- Confidence scores
- Detection status indicators

### Configuration
```python
VISUALIZE=True          # Enable/disable overlay
DISPLAY_FPS=True        # Show FPS counter
SHOW_BOX=True           # Show detection boxes
SHOW_TEXT=True          # Show plate text
FONT_SCALE=0.7          # Text size
BOX_COLOR=(0, 255, 0)   # Bounding box color (green)
TEXT_COLOR=(255, 255, 255)  # Text color (white)
```

Visualization must not affect inference performance — render in a separate thread.

## Configuration
Do not hard-code:
- model paths
- camera/RTSP URL
- GPU
- confidence/IoU
- OCR parameters
- callback URL
- timeout/retry
- cooldown
- visualization settings
- model test mode settings

Use environment variables and/or config files.

## 24/7 Reliability
- Camera disconnect → reconnect
- Bad frame → skip
- Inference error → log and continue
- Callback failure → retry/backoff
- SIGTERM → graceful shutdown
- Runtime crash → Docker restart

**Best-result capture logic:**
1. First detection of a new plate → start collecting up to 20 frames
2. Same plate detected → add to collection (up to 20 frames or timeout)
3. Collection complete → score all results → send best to server
4. Cooldown period → ignore same plate until expired

Use one inference worker by default to avoid duplicate model copies in VRAM.

## Docker
One container represents the LPR runtime:

```text
LPR Runtime Container
├── Plate Detector
├── OCR Text Detector
└── OCR Text Recognizer
```

Separate model modules do **not** mean separate containers.

Model weights should be versioned/mounted separately and not committed to Git.

Docker must support GPU, model volume, environment configuration, healthcheck, and `restart: unless-stopped`.

## Dependency Direction

```text
API
 ↓
Runtime Worker
 ↓
LPR Pipeline
 ↓
Model Interfaces

Camera → Runtime/Pipeline
Pipeline → Domain
Runtime → Events
Events → HTTP/Redis/etc.
```

Models must not depend on API, events, camera, or Docker.

## Testing

```text
tests/
├── unit/
├── integration/
└── smoke/
```

Mock models/external services for unit tests. Test real inference separately.

## Model Testing Mode

Run inference on camera feed or single image to visually verify all 3 models work correctly.

### Usage
```bash
python scripts/test_models.py --camera
python scripts/test_models.py --camera --camera-id 1
python scripts/test_models.py --image path/to/image.jpg
```

### Features
- **Stage 1:** YOLO plate detection — green boxes around plates
- **Stage 2:** OCR text detection — blue boxes around text regions
- **Stage 3:** OCR text recognition — red text overlaid on plates
- **Overlay:** Show each stage result on the frame for debugging
- **Stats:** Print timing per frame and total FPS

### Configuration
```python
TEST_IMAGE=None           # Path to test image
TEST_SHOW_STAGES=True     # Show intermediate results
TEST_SAVE_FRAME=False     # Save best frames to disk
```

### Quick Start - GPU Test
```bash
# 1. Cài môi trường
pip install -U paddleocr
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu124/
pip install -q -U ultralytics

# 2. Verify GPU
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"

# 3. Test với camera trực tiếp
python scripts/test_models.py --camera

# 4. Hoặc test với ảnh
python scripts/test_models.py --image path/to/image.jpg
```

### Interface
```python
class ModelTester:
    def load_models(self) -> None
    def run_on_camera(self, camera_id: int = 0) -> None
    def show_stage(self, stage: int, frame, detections) -> None
```

### GPU Verification Test
```python
# Quick GPU check script - chạy trước khi test models
import torch
import paddle

def verify_gpu():
    # Check CUDA
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
    
    # Check PaddlePaddle
    print(f"\nPaddlePaddle CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        print(f"PaddlePaddle GPU count: {paddle.device.cuda.device_count()}")
    
    # Check Ultralytics
    from ultralytics.utils.torch_utils import select_device
    device = select_device('0')
    print(f"\nUltralytics device: {device}")

verify_gpu()
```

## Environment Setup (GPU)

### Prerequisites
- NVIDIA GPU với CUDA support
- cuDNN installed
- Python 3.8+

### Installation Commands
```bash
# Cài đặt PaddleOCR
!pip install -U paddleocr

# Cài đặt PaddlePaddle GPU (CUDA 12.4)
!pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu124/

# Cài đặt Ultralytics (YOLO)
!pip install -q -U ultralytics
```

### Verify GPU Setup
```python
import paddle
print(f"PaddlePaddle version: {paddle.__version__}")
print(f"GPU available: {paddle.device.is_compiled_with_cuda()}")
print(f"GPU count: {paddle.device.cuda.device_count()}")

from paddle.device import get_device
print(f"Running on: {get_device()}")
```

### Test All Models
```python
# Test YOLO plate detector
from ultralytics import YOLO
plate_model = YOLO('models/plate_detector/Plate.pt')
results = plate_model('test_image.jpg', device='0')

# Test PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
text_results = ocr.ocr('cropped_plate.jpg')

# Test full pipeline
print("✅ All models loaded successfully on GPU")
```

## Implementation Order

1. Domain models
2. Logging setup
3. **GPU Environment Setup & Verification** ⬅️ **Test GPU setup first!**
4. Model adapters
5. Camera abstraction
6. Crop/preprocessing
7. LPR pipeline
8. Plate collector (collect 20 frames, pick best)
9. Event publisher
10. Runtime worker
11. API/health
12. **Model Testing Mode** ⬅️ **Verify all 3 models work on video**
13. Tests
14. Docker
15. 24/7 deployment

## Final Principle

```text
Camera
 ↓
Runtime
 ↓
Pipeline
 ↓
Models
 ↓
Domain Result
 ↓
Plate Collector (collect 20 frames)
 ↓
Best Result Selector (score & pick winner)
 ↓
Event Publisher
 ↓
APP Server
```

Do not create a single `app.py`, `utils.py`, or class containing the entire system.
