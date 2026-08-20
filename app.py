import os
import json
import time
import threading
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from paddleocr import TextDetection, TextRecognition

# ============================================================
# CONFIG
# ============================================================
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/models/Plate.pt")
YOLO_DEVICE = int(os.getenv("YOLO_DEVICE", "0"))
PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "gpu:0")

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
OCR_UPSCALE = int(os.getenv("OCR_UPSCALE", "4"))
TEXT_PADDING = int(os.getenv("TEXT_PADDING", "10"))
REC_MIN_SCORE = float(os.getenv("REC_MIN_SCORE", "0.0"))

# Camera / 24-7 worker
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
INFERENCE_FPS = float(os.getenv("INFERENCE_FPS", "5"))
CALLBACK_URL = os.getenv("CALLBACK_URL", "")
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "5"))
PLATE_COOLDOWN_SECONDS = float(os.getenv("PLATE_COOLDOWN_SECONDS", "30"))

# ============================================================
# APP STATE
# ============================================================
app = FastAPI(title="24/7 LPR Runtime", version="2.0.0")

worker_thread = None
worker_stop = threading.Event()
last_seen = {}
state_lock = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# MODEL LOADING
# ============================================================
print("Loading YOLO:", YOLO_MODEL_PATH)
plate_model = YOLO(YOLO_MODEL_PATH)

print("Loading PP-OCRv6_small_det...")
det_model = TextDetection(
    model_name="PP-OCRv6_small_det",
    device=PADDLE_DEVICE,
)

print("Loading PP-OCRv6_small_rec...")
rec_model = TextRecognition(
    model_name="PP-OCRv6_small_rec",
    device=PADDLE_DEVICE,
)

print("ALL MODELS LOADED")


# ============================================================
# OCR / DETECTION FUNCTIONS
# ============================================================
def get_result_dict(res):
    data = res.json
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data


def detect_plates(image):
    results = plate_model.predict(
        source=image,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device=YOLO_DEVICE,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()

    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        detections.append({
            "box": [x1, y1, x2, y2],
            "score": float(score),
            "class_id": int(cls),
            "class_name": plate_model.names[int(cls)],
        })

    return detections


def crop_plate(image, box, padding=0.05):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def upscale_plate(plate):
    return cv2.resize(
        plate,
        None,
        fx=OCR_UPSCALE,
        fy=OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )


def sort_text_boxes(polygons, scores):
    items = []
    for poly, score in zip(polygons, scores):
        poly = np.asarray(poly, dtype=np.float32)
        items.append({
            "polygon": poly,
            "score": float(score),
            "x_center": float(np.mean(poly[:, 0])),
            "y_center": float(np.mean(poly[:, 1])),
        })

    items.sort(key=lambda x: (x["y_center"], x["x_center"]))
    return items


def crop_polygon(image, polygon, padding=10):
    polygon = np.asarray(polygon, dtype=np.float32)

    x1 = int(np.floor(np.min(polygon[:, 0]))) - padding
    y1 = int(np.floor(np.min(polygon[:, 1]))) - padding
    x2 = int(np.ceil(np.max(polygon[:, 0]))) + padding
    y2 = int(np.ceil(np.max(polygon[:, 1]))) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.copyMakeBorder(
        crop, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def ocr_plate(plate):
    plate_up = upscale_plate(plate)

    det_results = list(
        det_model.predict(
            input=plate_up,
            batch_size=1,
        )
    )

    if not det_results:
        return []

    data = get_result_dict(det_results[0])
    polygons = data.get("dt_polys", [])
    scores = data.get("dt_scores", [])

    detections = sort_text_boxes(polygons, scores)

    crops = []
    metadata = []

    for i, item in enumerate(detections):
        crop = crop_polygon(
            plate_up,
            item["polygon"],
            padding=TEXT_PADDING,
        )
        if crop is None:
            continue

        crops.append(crop)
        metadata.append({
            "line": i,
            "det_score": item["score"],
            "polygon": item["polygon"].tolist(),
        })

    if not crops:
        return []

    rec_results = list(
        rec_model.predict(
            input=crops,
            batch_size=len(crops),
        )
    )

    outputs = []
    for meta, res in zip(metadata, rec_results):
        data = get_result_dict(res)

        text = str(data.get("rec_text", "")).strip()
        try:
            rec_score = float(data.get("rec_score", 0.0))
        except Exception:
            rec_score = 0.0

        if text and rec_score >= REC_MIN_SCORE:
            outputs.append({
                "line": meta["line"],
                "text": text,
                "det_score": meta["det_score"],
                "rec_score": rec_score,
                "polygon": meta["polygon"],
            })

    return outputs


def normalize_plate(text):
    # Normalize OCR output for comparison.
    # Keeps letters/numbers and removes spaces, dots, hyphens, etc.
    text = str(text).upper().strip()
    return "".join(ch for ch in text if ch.isalnum())


def process_frame(image):
    plates = detect_plates(image)
    output = []

    for plate_idx, detection in enumerate(plates):
        plate = crop_plate(
            image,
            detection["box"],
            padding=0.05,
        )
        if plate is None:
            continue

        ocr_results = ocr_plate(plate)

        # Keep OCR lines in reading order.
        final_text = "\n".join(
            item["text"] for item in ocr_results
        ).strip()

        normalized = normalize_plate(final_text)

        if not normalized:
            continue

        output.append({
            "plate_index": plate_idx,
            "plate": final_text,
            "plate_normalized": normalized,
            "box": detection["box"],
            "yolo_score": detection["score"],
            "class_name": detection["class_name"],
            "ocr": ocr_results,
        })

    return output


# ============================================================
# CALLBACK / REPORTING
# ============================================================
def should_report(plate):
    now = time.time()

    with state_lock:
        previous = last_seen.get(plate)

        if previous is not None:
            if now - previous < PLATE_COOLDOWN_SECONDS:
                return False

        last_seen[plate] = now

        # Prevent unbounded memory growth.
        cutoff = now - max(PLATE_COOLDOWN_SECONDS * 10, 300)
        stale = [p for p, t in last_seen.items() if t < cutoff]
        for p in stale:
            del last_seen[p]

    return True


def report_plate(result, camera_source):
    payload = {
        "event": "new_plate",
        "timestamp": utc_now(),
        "camera": str(camera_source),
        "plate": result["plate"],
        "plate_normalized": result["plate_normalized"],
        "yolo_score": result["yolo_score"],
        "box": result["box"],
        "ocr": result["ocr"],
    }

    print(
        f"[NEW PLATE] {payload['plate']} "
        f"score={payload['yolo_score']:.3f}"
    )

    if not CALLBACK_URL:
        print("[WARNING] CALLBACK_URL is not configured.")
        return False

    try:
        response = requests.post(
            CALLBACK_URL,
            json=payload,
            timeout=CALLBACK_TIMEOUT,
        )
        response.raise_for_status()
        print(
            f"[CALLBACK OK] status={response.status_code}"
        )
        return True
    except Exception as exc:
        print("[CALLBACK ERROR]", repr(exc))
        return False


# ============================================================
# 24/7 CAMERA WORKER
# ============================================================
def camera_worker():
    source = CAMERA_SOURCE

    try:
        source_for_cv = int(source)
    except ValueError:
        source_for_cv = source

    target_interval = 1.0 / max(INFERENCE_FPS, 0.1)

    print("=" * 70)
    print("STARTING 24/7 LPR CAMERA WORKER")
    print("Camera:", source)
    print("Inference FPS:", INFERENCE_FPS)
    print("Callback:", CALLBACK_URL or "(not configured)")
    print("Cooldown:", PLATE_COOLDOWN_SECONDS, "seconds")
    print("=" * 70)

    while not worker_stop.is_set():
        cap = None

        try:
            print("[CAMERA] Connecting...")
            cap = cv2.VideoCapture(source_for_cv)

            # Useful for network cameras; harmless for many local cameras.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[CAMERA] Cannot open camera. Retry in 5 seconds...")
                worker_stop.wait(5)
                continue

            print("[CAMERA] Connected.")

            last_inference = 0.0

            while not worker_stop.is_set():
                ok, frame = cap.read()

                if not ok or frame is None:
                    print("[CAMERA] Frame read failed. Reconnecting...")
                    break

                now = time.time()
                if now - last_inference < target_interval:
                    continue

                last_inference = now

                try:
                    results = process_frame(frame)

                    for result in results:
                        plate = result["plate_normalized"]

                        if should_report(plate):
                            report_plate(
                                result,
                                camera_source=source,
                            )

                except Exception as exc:
                    # Keep 24/7 worker alive even if one frame fails.
                    print("[INFERENCE ERROR]", repr(exc))

        except Exception as exc:
            print("[WORKER ERROR]", repr(exc))

        finally:
            if cap is not None:
                cap.release()

        if not worker_stop.is_set():
            print("[CAMERA] Reconnect in 3 seconds...")
            worker_stop.wait(3)

    print("[CAMERA] Worker stopped.")


# ============================================================
# API
# ============================================================
@app.on_event("startup")
def startup_event():
    global worker_thread

    worker_stop.clear()

    worker_thread = threading.Thread(
        target=camera_worker,
        name="lpr-camera-worker",
        daemon=True,
    )
    worker_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    worker_stop.set()

    if worker_thread is not None:
        worker_thread.join(timeout=5)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "worker_running": (
            worker_thread is not None
            and worker_thread.is_alive()
        ),
        "camera_source": CAMERA_SOURCE,
        "callback_url_configured": bool(CALLBACK_URL),
        "models": {
            "plate_detection": "Plate.pt",
            "ocr_detection": "PP-OCRv6_small_det",
            "ocr_recognition": "PP-OCRv6_small_rec",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Manual single-image test endpoint."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image uploads are supported.",
        )

    raw = await file.read()
    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode image.",
        )

    return {
        "plates": process_frame(image)
    }
