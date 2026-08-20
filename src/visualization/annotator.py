"""Visualization - Frame annotation and overlay.

Responsibilities (per PLAN.md):
- Draw boxes, text, FPS
- Must not block inference loop
- Run in separate thread
"""

import logging
from typing import Optional
import cv2
import numpy as np

from src.domain.models import LPRResult


logger = logging.getLogger("lpr.visualization.annotator")


class ResultAnnotator:
    """Annotate LPR results on frames."""

    def __init__(
        self,
        box_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.7,
        thickness: int = 2,
    ):
        """Initialize annotator.

        Args:
            box_color: Bounding box color (BGR)
            text_color: Text color (BGR)
            font_scale: Text scale
            thickness: Line thickness
        """
        self.box_color = box_color
        self.text_color = text_color
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_result(
        self,
        frame: np.ndarray,
        result: LPRResult,
    ) -> np.ndarray:
        """Draw LPR result on frame.

        Args:
            frame: Input frame
            result: LPR result

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw bounding box
        x1, y1, x2, y2 = result.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), self.box_color, self.thickness)

        # Draw plate text
        text = result.plate_normalized
        confidence = result.get_confidence()

        label = f"{text} ({confidence:.2f})"

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.thickness,
        )

        # Draw text background
        bg_y1 = max(y1 - text_h - 10, 0)
        bg_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, bg_y1),
            (x1 + text_w + 10, bg_y2),
            self.box_color,
            -1,  # Filled
        )

        # Draw text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.text_color,
            self.thickness,
        )

        return annotated

    def draw_results(
        self,
        frame: np.ndarray,
        results: list[LPRResult],
    ) -> np.ndarray:
        """Draw multiple results on frame.

        Args:
            frame: Input frame
            results: List of LPR results

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for result in results:
            annotated = self.draw_result(annotated, result)

        return annotated

    def draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
        position: str = "top-left",
    ) -> np.ndarray:
        """Draw FPS counter on frame.

        Args:
            frame: Input frame
            fps: FPS value
            position: Position (top-left, top-right, bottom-left, bottom-right)

        Returns:
            Frame with FPS
        """
        annotated = frame.copy()
        text = f"FPS: {fps:.1f}"

        (text_w, text_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
        )

        h, w = frame.shape[:2]

        # Calculate position
        if position == "top-left":
            x, y = 10, text_h + 10
        elif position == "top-right":
            x, y = w - text_w - 10, text_h + 10
        elif position == "bottom-left":
            x, y = 10, h - 10
        else:  # bottom-right
            x, y = w - text_w - 10, h - 10

        # Draw background
        cv2.rectangle(
            annotated,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + 5),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            annotated,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        return annotated


class PlateDetectionAnnotator:
    """Annotator for intermediate detection stages."""

    def __init__(
        self,
        plate_color: tuple[int, int, int] = (0, 255, 0),  # Green
        text_color_box: tuple[int, int, int] = (255, 0, 0),  # Blue
        rec_text_color: tuple[int, int, int] = (0, 0, 255),  # Red
    ):
        """Initialize stage annotator.

        Args:
            plate_color: Plate detection box color (green)
            text_color_box: Text detection box color (blue)
            rec_text_color: Recognized text color (red)
        """
        self.plate_color = plate_color
        self.text_color_box = text_color_box
        self.rec_text_color = rec_text_color

    def draw_plate_boxes(
        self,
        frame: np.ndarray,
        boxes: list,
        scores: list,
    ) -> np.ndarray:
        """Draw plate detection boxes.

        Args:
            frame: Input frame
            boxes: List of boxes [[x1,y1,x2,y2], ...]
            scores: List of confidence scores

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.plate_color, 2)
            cv2.putText(
                annotated,
                f"Plate: {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.plate_color,
                2,
            )

        return annotated

    def draw_text_boxes(
        self,
        frame: np.ndarray,
        polygons: list,
        texts: list = None,
    ) -> np.ndarray:
        """Draw text detection boxes.

        Args:
            frame: Input frame
            polygons: List of polygons
            texts: Optional list of recognized texts

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for i, poly in enumerate(polygons):
            poly = np.array(poly, dtype=np.int32)
            cv2.polylines(annotated, [poly], True, self.text_color_box, 2)

            if texts and i < len(texts):
                x, y = poly[0]
                cv2.putText(
                    annotated,
                    str(texts[i]),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.rec_text_color,
                    2,
                )

        return annotated
