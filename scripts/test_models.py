#!/usr/bin/env python3
"""
Test Models Script - Video-based model testing tool.

Usage:
    python scripts/test_models.py --video path/to/video.mp4 --output result.mp4
    python scripts/test_models.py --video path/to/video.mp4
    python scripts/test_models.py  # Uses camera if no video specified

Features:
    Stage 1: YOLO plate detection - green boxes around plates
    Stage 2: OCR text detection - blue boxes around text regions
    Stage 3: OCR text recognition - red text overlaid on plates
    Overlay: Show each stage result on the frame for debugging
    Save: Optional output video with all annotations
    Stats: Print timing per stage and total FPS
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddleocr import TextDetection, TextRecognition
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

# Default model paths (can be overridden via environment variables)
DEFAULT_YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Plate.pt")
)

DEFAULT_PADDLE_MODEL_DIR = os.getenv(
    "PADDLE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
)

# Default image directory for testing
DEFAULT_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Test_Images")


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

def get_device_config():
    """Get device configuration for YOLO and PaddleOCR.

    Returns:
        Tuple of (yolo_device, paddle_device)
    """
    # YOLO / Ultralytics uses:
    #   0       -> GPU 0
    #   1       -> GPU 1
    #   "cpu"   -> CPU
    yolo_device = int(os.getenv("YOLO_DEVICE", "0"))

    # PaddleOCR uses:
    #   "gpu:0"
    #   "gpu:1"
    #   "cpu"
    paddle_device = os.getenv("PADDLE_DEVICE", "gpu:0")

    return yolo_device, paddle_device


# ============================================================
# PARAMETERS
# ============================================================

# YOLO parameters
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))

# OCR parameters
OCR_UPSCALE = int(os.getenv("OCR_UPSCALE", "4"))
TEXT_PADDING = int(os.getenv("TEXT_PADDING", "10"))
REC_MIN_SCORE = float(os.getenv("REC_MIN_SCORE", "0.0"))


# ============================================================
# MODEL LOADING
# ============================================================

def print_device_info():
    """Print CUDA device information."""
    print("=" * 70)
    print("DEVICE CONFIGURATION")
    print("=" * 70)

    yolo_device, paddle_device = get_device_config()
    print(f"YOLO device    : {yolo_device}")
    print(f"Paddle device  : {paddle_device}")

    try:
        import torch
        print()
        print(f"PyTorch CUDA available : {torch.cuda.is_available()}")
        print(f"CUDA device count      : {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Could not inspect PyTorch CUDA: {e}")

    print("=" * 70)


def load_models():
    """Load all ML models.

    Returns:
        Tuple of (plate_model, det_model, rec_model)
    """
    yolo_device, paddle_device = get_device_config()

    # Load YOLO plate detector
    print(f"\nLoading YOLO plate detector from {DEFAULT_YOLO_MODEL_PATH}...")
    plate_model = YOLO(DEFAULT_YOLO_MODEL_PATH)
    print(f"YOLO classes: {plate_model.names}")
    print(f"YOLO device: {yolo_device}")
    print("YOLO loaded successfully!")

    # Load PaddleOCR text detector
    det_model_path = os.path.join(DEFAULT_PADDLE_MODEL_DIR, "PP-OCRv6_small_det")
    print(f"\nLoading PP-OCRv6_small_det from {det_model_path}...")
    det_model = TextDetection(
        model_name="PP-OCRv6_small_det",
        model_dir=det_model_path,
        device=paddle_device
    )
    print(f"PaddleOCR DET device: {paddle_device}")
    print("PP-OCRv6_small_det loaded!")

    # Load PaddleOCR text recognizer
    rec_model_path = os.path.join(DEFAULT_PADDLE_MODEL_DIR, "PP-OCRv6_small_rec")
    print(f"\nLoading PP-OCRv6_small_rec from {rec_model_path}...")
    rec_model = TextRecognition(
        model_name="PP-OCRv6_small_rec",
        model_dir=rec_model_path,
        device=paddle_device
    )
    print(f"PaddleOCR REC device: {paddle_device}")
    print("PP-OCRv6_small_rec loaded!")

    print("\n" + "=" * 70)
    print("ALL MODELS LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"YOLO       -> GPU {yolo_device}")
    print(f"PaddleOCR  -> {paddle_device}")
    print("=" * 70)

    return plate_model, det_model, rec_model


# ============================================================
# RESULT PARSING
# ============================================================

def get_result_dict(res):
    """Parse PaddleOCR result to extract detection data."""
    data = res.json
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_plates(image, model, conf=YOLO_CONF, iou=YOLO_IOU):
    """Detect plates using YOLO.

    Args:
        image: Input image (BGR format)
        model: YOLO model
        conf: Confidence threshold
        iou: IoU threshold

    Returns:
        List of detection dictionaries
    """
    yolo_device, _ = get_device_config()

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        device=yolo_device,
        verbose=False
    )

    if not results or not results[0].boxes:
        return []

    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
    scores = results[0].boxes.conf.detach().cpu().numpy()
    classes = results[0].boxes.cls.detach().cpu().numpy()

    detections = []
    for box, score, cls in zip(boxes, scores, classes, strict=True):
        x1, y1, x2, y2 = map(int, box)
        detections.append({
            "box": [x1, y1, x2, y2],
            "score": float(score),
            "class_id": int(cls),
            "class_name": model.names[int(cls)],
        })

    return detections


def crop_plate(image, box, padding=0.05):
    """Crop plate from image with padding.

    Args:
        image: Input image
        box: Bounding box [x1, y1, x2, y2]
        padding: Padding ratio

    Returns:
        Cropped plate image or None
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def upscale_plate(plate, scale=OCR_UPSCALE):
    """Upscale plate image for OCR.

    Args:
        plate: Plate image
        scale: Upscale factor

    Returns:
        Upscaled plate image
    """
    if plate is None or plate.size == 0:
        return None

    return cv2.resize(
        plate,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


# ============================================================
# OCR DETECTION
# ============================================================

def detect_text(plate, model, scale=OCR_UPSCALE):
    """Detect text regions using PaddleOCR.

    Args:
        plate: Plate image (already upscaled)
        model: Text detection model
        scale: Original upscale factor

    Returns:
        Tuple of (upscaled_plate, detections)
    """
    plate_up = upscale_plate(plate, scale)
    if plate_up is None:
        return None, []

    results = list(model.predict(input=plate_up, batch_size=1))
    if not results:
        return plate_up, []

    data = get_result_dict(results[0])
    polygons = data.get("dt_polys", [])
    scores = data.get("dt_scores", [])

    detections = []
    for poly, score in zip(polygons, scores, strict=True):
        polygon = np.asarray(poly, dtype=np.float32)
        detections.append({
            "polygon": polygon,
            "score": float(score),
            "x_center": float(np.mean(polygon[:, 0])),
            "y_center": float(np.mean(polygon[:, 1])),
        })

    # Sort by reading order (top -> bottom, left -> right)
    detections.sort(key=lambda x: (x["y_center"], x["x_center"]))

    return plate_up, detections


def crop_polygon(image, polygon, padding=TEXT_PADDING):
    """Crop polygon region from image.

    Args:
        image: Input image
        polygon: Polygon coordinates
        padding: Padding in pixels

    Returns:
        Cropped polygon image
    """
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

    return image[y1:y2, x1:x2]


def recognize_text(crops, model):
    """Recognize text from cropped images.

    Args:
        crops: List of cropped images
        model: Text recognition model

    Returns:
        List of recognition results
    """
    if not crops:
        return []

    results = list(model.predict(input=crops, batch_size=len(crops)))
    outputs = []

    for res in results:
        data = get_result_dict(res)
        text = data.get("rec_text", "").strip()
        try:
            score = float(data.get("rec_score", 0.0))
        except Exception:
            score = 0.0

        outputs.append({
            "text": text,
            "score": score,
        })

    return outputs


def ocr_plate(plate, det_model, rec_model, visualize=True):
    """Run full OCR pipeline on a plate.

    Args:
        plate: Cropped plate image
        det_model: Text detection model
        rec_model: Text recognition model
        visualize: Whether to show visualizations

    Returns:
        List of OCR results
    """
    if plate is None or plate.size == 0:
        return []

    print("\n" + "-" * 50)
    print("OCR DETECTION")
    print("-" * 50)
    print(f"Original shape: {plate.shape}")

    # Detect text
    plate_up, detections = detect_text(plate, det_model)
    if plate_up is None:
        return []

    print(f"Upscaled shape: {plate_up.shape}")
    print(f"Text boxes detected: {len(detections)}")

    if not detections:
        print("NO TEXT DETECTED")
        if visualize:
            plt.figure(figsize=(12, 5))
            plt.imshow(cv2.cvtColor(plate_up, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("PP-OCRv6_small_det - NO TEXT")
            plt.show()
        return []

    for i, item in enumerate(detections):
        print(f"Box {i}: det_score={item['score']:.4f}")

    # Crop text regions
    crops = []
    metadata = []

    for i, item in enumerate(detections):
        crop = crop_polygon(plate_up, item["polygon"], padding=TEXT_PADDING)
        if crop is None or crop.size == 0:
            continue

        # Add white padding
        crop = cv2.copyMakeBorder(
            crop, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        crops.append(crop)
        metadata.append({
            "line": i,
            "det_score": item["score"],
            "polygon": item["polygon"],
        })

    # Visualize text crops
    if visualize and crops:
        _, axes = plt.subplots(1, len(crops), figsize=(6 * len(crops), 4))
        if len(crops) == 1:
            axes = [axes]
        for i, crop in enumerate(crops):
            axes[i].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[i].axis("off")
            axes[i].set_title(f"Line {i}")
        plt.tight_layout()
        plt.show()

    # Recognize text
    results = recognize_text(crops, rec_model)

    print("\n" + "-" * 50)
    print("OCR RECOGNITION")
    print("-" * 50)

    outputs = []
    for meta, rec in zip(metadata, results, strict=True):
        if rec["text"] and rec["score"] >= REC_MIN_SCORE:
            outputs.append({
                "line": meta["line"],
                "text": rec["text"],
                "det_score": meta["det_score"],
                "rec_score": rec["score"],
                "polygon": meta["polygon"],
            })
            print(f"Line {meta['line']}: '{rec['text']}' | DET={meta['det_score']:.4f} | REC={rec['score']:.4f}")

    print("\n" + "-" * 50)
    print("FINAL PLATE TEXT")
    print("-" * 50)
    final_text = " ".join(r["text"] for r in outputs)
    print(final_text if final_text else "No recognized text.")

    return outputs


# ============================================================
# VISUALIZATION
# ============================================================

def draw_plate_detections(image, detections, show=True):
    """Draw plate detection boxes.

    Args:
        image: Input image
        detections: List of detections
        show: Whether to display

    Returns:
        Annotated image
    """
    vis = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"plate {score:.2f}"
        cv2.putText(vis, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if show:
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("YOLO Plate Detection")
        plt.show()

    return vis


def draw_ocr_detections(image, detections, show=True):
    """Draw OCR detection boxes.

    Args:
        image: Input image
        detections: List of detections
        show: Whether to display

    Returns:
        Annotated image
    """
    vis = image.copy()

    for i, item in enumerate(detections):
        poly = np.asarray(item["polygon"], dtype=np.int32)
        cv2.polylines(vis, [poly], True, (255, 0, 0), 2)

        x = int(np.min(poly[:, 0]))
        y = int(np.min(poly[:, 1]))
        cv2.putText(vis, f"{i}: {item['score']:.2f}", (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    if show:
        plt.figure(figsize=(14, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("PP-OCRv6_small_det")
        plt.show()

    return vis


def draw_final_results(plate, ocr_results, show=True):
    """Draw final OCR results on plate.

    Args:
        plate: Plate image (upscaled)
        ocr_results: List of OCR results
        show: Whether to display

    Returns:
        Annotated image
    """
    vis = plate.copy()

    for r in ocr_results:
        poly = np.asarray(r["polygon"], dtype=np.int32)

        # Draw box
        cv2.polylines(vis, [poly], True, (0, 0, 255), 2)

        # Draw text
        x = int(np.min(poly[:, 0]))
        y = int(np.max(poly[:, 1])) + 30
        cv2.putText(vis, r["text"], (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    if show:
        plt.figure(figsize=(12, 4))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Final OCR Results")
        plt.show()

    return vis


# ============================================================
# PROCESSING
# ============================================================

def process_image(image_path, plate_model, det_model, rec_model, visualize=True):
    """Process a single image.

    Args:
        image_path: Path to image
        plate_model: YOLO plate model
        det_model: PaddleOCR text detector
        rec_model: PaddleOCR text recognizer
        visualize: Whether to show visualizations

    Returns:
        List of processing results
    """
    print("\n" + "=" * 70)
    print(f"IMAGE: {os.path.basename(image_path)}")
    print("=" * 70)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    print(f"Image shape: {image.shape}")

    # YOLO detection
    print(f"\nRunning YOLO on GPU {get_device_config()[0]}...")
    plates = detect_plates(image, plate_model)
    print(f"\nYOLO detected plates: {len(plates)}")

    if not plates:
        print("YOLO did not detect any plate.")
        if visualize:
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("YOLO - NO PLATE DETECTED")
            plt.show()
        return []

    if visualize:
        draw_plate_detections(image, plates)

    all_results = []

    for plate_idx, plate_det in enumerate(plates):
        print(f"\n{'#' * 60}")
        print(f"PLATE {plate_idx}")
        print(f"{'#' * 60}")
        print(f"YOLO confidence: {plate_det['score']:.4f}")
        print(f"YOLO class: {plate_det['class_name']}")
        print(f"YOLO box: {plate_det['box']}")

        # Crop plate
        plate = crop_plate(image, plate_det['box'], padding=0.05)
        if plate is None:
            print("Plate crop failed.")
            continue

        print(f"Plate crop shape: {plate.shape}")

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"YOLO Plate Crop {plate_idx}")
            plt.show()

        # Run OCR
        print(f"\nRunning PaddleOCR on {get_device_config()[1]}...")
        ocr_results = ocr_plate(plate, det_model, rec_model, visualize=visualize)

        all_results.append({
            "plate_index": plate_idx,
            "yolo": plate_det,
            "ocr": ocr_results
        })

    return all_results


def process_video(video_path, output_path, plate_model, det_model, rec_model):
    """Process video file.

    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        plate_model: YOLO plate model
        det_model: PaddleOCR text detector
        rec_model: PaddleOCR text recognizer
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\nVideo: {width}x{height} @ {fps} FPS")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")

    frame_count = 0
    total_plates = 0

    print("\n" + "=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        # Detect plates
        plates = detect_plates(frame, plate_model, visualize=False)

        for plate_det in plates:
            x1, y1, x2, y2 = plate_det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Crop and OCR
            plate = crop_plate(frame, plate_det['box'], padding=0.05)
            if plate is not None:
                plate_up, detections = detect_text(plate, det_model, visualize=False)

                if detections:
                    # Crop and recognize
                    crops = []
                    metadata = []
                    for item in detections:
                        crop = crop_polygon(plate_up, item["polygon"], padding=TEXT_PADDING)
                        if crop is not None:
                            crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10,
                                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
                            crops.append(crop)
                            metadata.append(item)

                    if crops:
                        results = recognize_text(crops, rec_model)
                        for meta, rec in zip(metadata, results, strict=True):
                            if rec["text"]:
                                # Draw text on frame
                                text_x = x1 + int(np.min(meta["polygon"][:, 0]) / OCR_UPSCALE)
                                text_y = y2
                                cv2.putText(frame, rec["text"], (text_x, text_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                total_plates += 1

        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Plates: {len(plates)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        # Display
        cv2.imshow("LPR Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total frames: {frame_count}")
    print(f"Total plates detected: {total_plates}")
    print(f"Average plates/frame: {total_plates / max(frame_count, 1):.2f}")


def process_camera(plate_model, det_model, rec_model):
    """Process camera feed.

    Args:
        plate_model: YOLO plate model
        det_model: PaddleOCR text detector
        rec_model: PaddleOCR text recognizer
    """
    camera_source = int(os.getenv("CAMERA_SOURCE", "0"))
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {camera_source}")

    print(f"\nCamera opened: {camera_source}")

    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed, retrying...")
            continue

        frame_count += 1

        # Detect plates
        plates = detect_plates(frame, plate_model, visualize=False)

        for plate_det in plates:
            x1, y1, x2, y2 = plate_det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Crop and OCR
            plate = crop_plate(frame, plate_det['box'], padding=0.05)
            if plate is not None:
                plate_up, detections = detect_text(plate, det_model, visualize=False)

                if detections:
                    crops = []
                    metadata = []
                    for item in detections:
                        crop = crop_polygon(plate_up, item["polygon"], padding=TEXT_PADDING)
                        if crop is not None:
                            crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10,
                                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
                            crops.append(crop)
                            metadata.append(item)

                    if crops:
                        results = recognize_text(crops, rec_model)
                        for meta, rec in zip(metadata, results, strict=True):
                            if rec["text"]:
                                text_x = x1 + int(np.min(meta["polygon"][:, 0]) / OCR_UPSCALE)
                                text_y = y2
                                cv2.putText(frame, rec["text"], (text_x, text_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Info overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Plates: {len(plates)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("LPR Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LPR Model Testing Tool")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--camera", action="store_true", help="Use camera")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualizations")
    parser.add_argument("--yolo-model", type=str, default=DEFAULT_YOLO_MODEL_PATH,
                       help="Path to YOLO model")
    parser.add_argument("--paddle-model-dir", type=str, default=DEFAULT_PADDLE_MODEL_DIR,
                       help="Directory containing PaddleOCR models")

    args = parser.parse_args()

    # Update global paths
    global DEFAULT_YOLO_MODEL_PATH, DEFAULT_PADDLE_MODEL_DIR
    DEFAULT_YOLO_MODEL_PATH = args.yolo_model
    DEFAULT_PADDLE_MODEL_DIR = args.paddle_model_dir

    visualize = not args.no_visualize

    # Print device info
    print_device_info()

    # Load models
    plate_model, det_model, rec_model = load_models()

    # Process based on input type
    if args.image:
        process_image(args.image, plate_model, det_model, rec_model, visualize=visualize)
    elif args.video:
        process_video(args.video, args.output, plate_model, det_model, rec_model)
    elif args.camera:
        process_camera(plate_model, det_model, rec_model)
    else:
        # Try to find test image
        test_image = os.path.join(DEFAULT_IMG_DIR, "CarLongPlate0_jpg.rf.e861e8ff15501637fc82e10ffb5299c0.jpg")
        if os.path.exists(test_image):
            process_image(test_image, plate_model, det_model, rec_model, visualize=visualize)
        else:
            print("\nNo input specified. Available options:")
            print("  --image <path>    Process single image")
            print("  --video <path>     Process video file")
            print("  --camera           Use camera feed")
            print("\nOr specify path to test image:")
            print(f"  python scripts/test_models.py --image {test_image}")


if __name__ == "__main__":
    main()
