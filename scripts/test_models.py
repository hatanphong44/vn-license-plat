"""Model Testing Script.

Per PLAN.md: Run inference on a video file to visually verify all 3 models work correctly.

Usage:
    python scripts/test_models.py --video path/to/video.mp4 --output result.mp4
    python scripts/test_models.py  # Use camera
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    YOLOPlateDetector,
    PaddleTextDetector,
    PaddleTextRecognizer,
)
from src.pipeline.cropper import PlateCropper, TextCropper, PlatePreprocessor
from src.visualization.annotator import ResultAnnotator, PlateDetectionAnnotator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_models")


class ModelTester:
    """Test all 3 models on video or camera."""

    def __init__(
        self,
        yolo_path: str,
        paddle_device: str = "gpu:0",
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        upscale: int = 4,
        show_stages: bool = True,
    ):
        """Initialize model tester.

        Args:
            yolo_path: Path to YOLO model
            paddle_device: Device for PaddleOCR
            yolo_conf: YOLO confidence threshold
            yolo_iou: YOLO IoU threshold
            upscale: Upscale factor for OCR
            show_stages: Show intermediate results
        """
        self.yolo_path = yolo_path
        self.paddle_device = paddle_device
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.upscale = upscale
        self.show_stages = show_stages

        self._plate_detector = None
        self._text_detector = None
        self._text_recognizer = None
        self._plate_cropper = PlateCropper(padding=0.05)
        self._text_cropper = TextCropper(padding=10)
        self._preprocessor = PlatePreprocessor(upscale_factor=upscale)
        self._annotator = ResultAnnotator()
        self._stage_annotator = PlateDetectionAnnotator()

    def load_models(self) -> None:
        """Load all models."""
        logger.info("=" * 60)
        logger.info("LOADING MODELS")
        logger.info("=" * 60)

        # Stage 1: YOLO Plate Detector
        logger.info(f"Loading YOLO plate detector: {self.yolo_path}")
        self._plate_detector = YOLOPlateDetector(
            model_path=self.yolo_path,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            device="0" if "gpu" in self.paddle_device else "cpu",
        )
        self._plate_detector.load()
        logger.info("✓ YOLO plate detector loaded")

        # Stage 2: PaddleOCR Text Detector
        logger.info("Loading PP-OCRv6_small_det...")
        self._text_detector = PaddleTextDetector(device=self.paddle_device)
        self._text_detector.load()
        logger.info("✓ OCR text detector loaded")

        # Stage 3: PaddleOCR Text Recognizer
        logger.info("Loading PP-OCRv6_small_rec...")
        self._text_recognizer = PaddleTextRecognizer(device=self.paddle_device)
        self._text_recognizer.load()
        logger.info("✓ OCR text recognizer loaded")

        logger.info("=" * 60)
        logger.info("ALL MODELS LOADED SUCCESSFULLY")
        logger.info("=" * 60)

    def run_on_video(
        self,
        video_path: str,
        output_path: str = None,
        save_frames: bool = False,
    ) -> None:
        """Run inference on video file.

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            save_frames: Save individual frames
        """
        logger.info(f"Opening video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        # Setup output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Output video: {output_path}")

        # Processing loop
        frame_num = 0
        stage_times = {"plate": [], "text_det": [], "text_rec": [], "total": []}

        logger.info("Starting inference...")
        logger.info("Press 'q' to quit, 's' to save current frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video")
                break

            frame_num += 1
            loop_start = time.time()

            # Run inference
            result_frame, times = self._process_frame(frame)
            stage_times["total"].append(time.time() - loop_start)

            for k, v in times.items():
                if k in stage_times:
                    stage_times[k].append(v)

            # Calculate FPS
            elapsed = time.time() - loop_start
            current_fps = 1.0 / elapsed if elapsed > 0 else 0

            # Draw FPS
            cv2.putText(
                result_frame,
                f"FPS: {current_fps:.1f} | Frame: {frame_num}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("LPR Model Test", result_frame)

            # Write output
            if writer:
                writer.write(result_frame)

            # Save frame on 's'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') and save_frames:
                save_path = f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(save_path, result_frame)
                logger.info(f"Saved: {save_path}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print stats
        self._print_stats(stage_times, frame_num)

    def run_on_camera(self, camera_id: int = 0) -> None:
        """Run inference on live camera.

        Args:
            camera_id: Camera device ID
        """
        logger.info(f"Opening camera {camera_id}")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        logger.info("Camera opened. Starting inference...")
        logger.info("Press 'q' to quit")

        frame_num = 0
        stage_times = {"plate": [], "text_det": [], "text_rec": [], "total": []}

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed")
                continue

            frame_num += 1
            loop_start = time.time()

            # Run inference
            result_frame, times = self._process_frame(frame)
            stage_times["total"].append(time.time() - loop_start)

            for k, v in times.items():
                if k in stage_times:
                    stage_times[k].append(v)

            # FPS
            elapsed = time.time() - loop_start
            current_fps = 1.0 / elapsed if elapsed > 0 else 0

            cv2.putText(
                result_frame,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("LPR Model Test", result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self._print_stats(stage_times, frame_num)

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Process a single frame.

        Args:
            frame: Input frame

        Returns:
            Tuple of (annotated frame, stage times dict)
        """
        times = {}
        result_frame = frame.copy()

        # Stage 1: Plate Detection (YOLO)
        t0 = time.time()
        plates = self._plate_detector.detect(frame)
        times["plate"] = time.time() - t0

        if not plates:
            return result_frame, times

        # Draw plate boxes (green) if showing stages
        if self.show_stages:
            boxes = [p.box for p in plates]
            scores = [p.score for p in plates]
            result_frame = self._stage_annotator.draw_plate_boxes(
                result_frame, boxes, scores
            )

        # Process each plate
        all_ocr_results = []

        for plate_idx, plate_det in enumerate(plates):
            # Crop plate
            plate_crop = self._plate_cropper.crop(frame, plate_det)
            if plate_crop is None:
                continue

            # Preprocess
            plate_prep = self._preprocessor.preprocess(plate_crop)

            # Stage 2: Text Detection
            t1 = time.time()
            text_dets = self._text_detector.detect(plate_prep)
            times["text_det"] = (times.get("text_det", 0) + time.time() - t1) / 2

            if not text_dets:
                continue

            # Draw text boxes (blue) if showing stages
            if self.show_stages:
                polys = [d.polygon.tolist() for d in text_dets]
                result_frame = self._stage_annotator.draw_text_boxes(
                    result_frame, polys
                )

            # Crop and recognize text
            crops = []
            for det in text_dets:
                crop = self._text_cropper.crop(plate_prep, det)
                if crop is not None:
                    crops.append(crop)

            if not crops:
                continue

            # Stage 3: Text Recognition
            t2 = time.time()
            texts = self._text_recognizer.recognize(crops)
            times["text_rec"] = (times.get("text_rec", 0) + time.time() - t2) / 2

            if texts:
                all_ocr_results.extend(texts)

        # Final annotation: draw recognized text (red) on result
        if all_ocr_results:
            # Create a simple result object for drawing
            from src.domain.models import LPRResult

            result = LPRResult(
                plate_index=0,
                plate=" | ".join(t.text for t in all_ocr_results),
                plate_normalized="".join(t.text for t in all_ocr_results).replace("_", ""),
                box=plates[0].box if plates else [0, 0, 100, 100],
                yolo_score=plates[0].score if plates else 0,
                class_name="plate",
                ocr_results=all_ocr_results,
            )

            result_frame = self._annotator.draw_result(result_frame, result)

        return result_frame, times

    def _print_stats(self, times: dict, total_frames: int) -> None:
        """Print processing statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {total_frames}")

        for stage, stage_times in times.items():
            if stage_times:
                avg_time = sum(stage_times) / len(stage_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                logger.info(f"{stage:12s}: avg={avg_time*1000:.1f}ms, fps={fps:.1f}")

        if times.get("total"):
            total_avg = sum(times["total"]) / len(times["total"])
            logger.info(f"{'TOTAL':12s}: avg={total_avg*1000:.1f}ms, "
                       f"fps={1.0/total_avg:.1f}")


def verify_gpu():
    """Verify GPU setup."""
    logger.info("=" * 60)
    logger.info("GPU VERIFICATION")
    logger.info("=" * 60)

    # PyTorch
    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not installed")

    # PaddlePaddle
    try:
        import paddle
        logger.info(f"PaddlePaddle compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")
        if paddle.device.is_compiled_with_cuda():
            logger.info(f"PaddlePaddle GPU count: {paddle.device.cuda.device_count()}")
    except ImportError:
        logger.warning("PaddlePaddle not installed")

    # Ultralytics
    try:
        from ultralytics.utils.torch_utils import select_device
        try:
            device = select_device('0')
        except ValueError:
            device = select_device('cpu')
        logger.info(f"Ultralytics device: {device}")
    except ImportError:
        logger.warning("Ultralytics not installed")

    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test LPR models on video or camera"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output video file"
    )
    parser.add_argument(
        "--yolo", "-y",
        type=str,
        default="/models/plate_detector/Plate.pt",
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="Device for PaddleOCR (gpu:0, cpu)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold"
    )
    parser.add_argument(
        "--upscale", "-u",
        type=int,
        default=4,
        help="OCR upscale factor"
    )
    parser.add_argument(
        "--no-stages",
        action="store_true",
        help="Don't show intermediate stages"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames on 's' key"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID (use with --camera flag)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify GPU setup only"
    )

    args = parser.parse_args()

    # Verify GPU if requested
    if args.verify:
        verify_gpu()
        return

    # Resolve YOLO path
    if not os.path.isabs(args.yolo):
        # Relative to script location
        script_dir = Path(__file__).parent.parent
        yolo_path = script_dir / args.yolo
    else:
        yolo_path = Path(args.yolo)

    if not yolo_path.exists():
        logger.error(f"YOLO model not found: {yolo_path}")
        return

    # Create tester
    tester = ModelTester(
        yolo_path=str(yolo_path),
        paddle_device=args.device,
        yolo_conf=args.conf,
        yolo_iou=args.iou,
        upscale=args.upscale,
        show_stages=not args.no_stages,
    )

    # Load models
    tester.load_models()

    # Run inference
    if args.video:
        tester.run_on_video(
            video_path=args.video,
            output_path=args.output,
            save_frames=args.save_frames,
        )
    else:
        tester.run_on_camera(camera_id=args.camera)


if __name__ == "__main__":
    main()
