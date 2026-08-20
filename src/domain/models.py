"""Domain models for LPR Runtime.

Defines all typed domain objects used across the application.
Following PLAN.md: PlateDetection, TextDetection, TextRecognition, LPRResult,
CapturedPlate, PlateCollection, PlateEvent.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np


@dataclass
class PlateDetection:
    """YOLO plate detection result."""
    box: list[int]  # [x1, y1, x2, y2]
    score: float  # YOLO confidence
    class_id: int
    class_name: str


@dataclass
class TextDetection:
    """OCR text detection result (bounding box polygon)."""
    polygon: np.ndarray  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    score: float  # detection confidence
    x_center: float
    y_center: float

    @classmethod
    def from_polygon_score(cls, polygon: np.ndarray, score: float) -> "TextDetection":
        return cls(
            polygon=polygon,
            score=float(score),
            x_center=float(np.mean(polygon[:, 0])),
            y_center=float(np.mean(polygon[:, 1])),
        )


@dataclass
class TextRecognition:
    """OCR text recognition result."""
    text: str
    line: int  # reading order (0 = first line)
    det_score: float  # detection confidence
    rec_score: float  # recognition confidence
    polygon: list  # normalized polygon coords


@dataclass
class LPRResult:
    """Full LPR pipeline result for a single plate detection."""
    plate_index: int
    plate: str  # raw OCR text (may have multiple lines)
    plate_normalized: str  # normalized (letters/numbers only, uppercase)
    box: list[int]  # [x1, y1, x2, y2] in frame coords
    yolo_score: float
    class_name: str
    ocr_results: list[TextRecognition]
    frame: np.ndarray | None = None  # Optional: full frame where detected

    def has_text(self) -> bool:
        """Check if plate has valid text."""
        return bool(self.plate_normalized)

    def get_confidence(self) -> float:
        """Get combined confidence score."""
        if not self.ocr_results:
            return self.yolo_score
        avg_rec_score = sum(r.rec_score for r in self.ocr_results) / len(self.ocr_results)
        return (self.yolo_score + avg_rec_score) / 2


@dataclass
class CapturedPlate:
    """A single plate detection captured during collection phase."""
    plate_normalized: str
    plate: str
    confidence: float  # OCR confidence
    yolo_score: float
    box: list[int]
    ocr_results: list[TextRecognition]
    frame: np.ndarray | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    frames_count: int = 0  # Number of frames used to collect this result


@dataclass
class PlateCollection:
    """Collection of frames for the same plate number."""
    plate_number: str  # normalized plate number
    captures: list[CapturedPlate] = field(default_factory=list)
    max_frames: int = 20
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add(self, capture: CapturedPlate) -> None:
        """Add a capture to collection if not full."""
        if len(self.captures) < self.max_frames:
            self.captures.append(capture)

    def is_full(self) -> bool:
        """Check if collection reached max frames."""
        return len(self.captures) >= self.max_frames

    def size(self) -> int:
        """Number of captures in collection."""
        return len(self.captures)

    def should_timeout(self, max_wait_seconds: float) -> bool:
        """Check if collection timed out waiting for more frames."""
        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
        return elapsed >= max_wait_seconds

    def get_text_counts(self) -> dict[str, int]:
        """Count occurrences of each plate text."""
        from collections import Counter
        return Counter(c.plate_normalized for c in self.captures)

    def get_best_result(self) -> CapturedPlate | None:
        """Get the best result using majority vote + confidence scoring."""
        if not self.captures:
            return None

        text_counts = self.get_text_counts()
        if not text_counts:
            return None

        # Get most common text
        most_common_text, max_votes = text_counts.most_common(1)[0]

        # Filter captures matching the most common text
        best_captures = [c for c in self.captures if c.plate_normalized == most_common_text]

        # Score each capture
        scored = []
        for cap in best_captures:
            score = (
                cap.confidence * 0.4 +  # OCR confidence
                cap.yolo_score * 0.3 +  # YOLO confidence
                len(cap.plate_normalized) * 0.1 +  # Longer text = better
                max_votes * 0.2  # More votes = better
            )
            scored.append((score, cap))

        # Return highest scored capture
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None


@dataclass
class PlateEvent:
    """Event to be published when plate collection completes."""
    event_type: str = "new_plate"
    timestamp: str = ""  # ISO format
    camera: str = ""
    plate: str = ""
    plate_normalized: str = ""
    yolo_score: float = 0.0
    box: list[int] = field(default_factory=list)
    ocr: list = field(default_factory=list)
    frames_collected: int = 0
    best_confidence: float = 0.0

    @classmethod
    def from_result(cls, result: CapturedPlate, camera: str, frames_count: int) -> "PlateEvent":
        """Create event from best LPR result."""
        return cls(
            timestamp=datetime.now(UTC).isoformat(),
            camera=str(camera),
            plate=result.plate,
            plate_normalized=result.plate_normalized,
            yolo_score=result.yolo_score,
            box=result.box,
            ocr=[{"text": r.text, "score": r.rec_score} for r in result.ocr_results],
            frames_collected=frames_count,
            best_confidence=result.confidence,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event_type,
            "timestamp": self.timestamp,
            "camera": self.camera,
            "plate": self.plate,
            "plate_normalized": self.plate_normalized,
            "yolo_score": self.yolo_score,
            "box": self.box,
            "ocr": self.ocr,
            "frames_collected": self.frames_collected,
            "best_confidence": self.best_confidence,
        }
