"""Unit tests for postprocessor."""


from src.domain.models import TextRecognition
from src.pipeline.postprocessor import (
    LPRPostProcessor,
    MultiLineConcatenator,
    TextNormalizer,
)


class TestTextNormalizer:
    """Test TextNormalizer."""

    def test_normalize_uppercase(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("abc123") == "ABC123"

    def test_normalize_removes_special_chars(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("AB.12-34") == "AB1234"
        assert normalizer.normalize("A B C 1 2 3") == "ABC123"

    def test_is_valid(self):
        normalizer = TextNormalizer(min_length=4, max_length=10)
        assert normalizer.is_valid("ABC123") is True
        assert normalizer.is_valid("ABC") is False  # too short
        assert normalizer.is_valid("ABCDEFGHIJKL") is False  # too long

    def test_clean_results(self):
        normalizer = TextNormalizer()
        results = [
            TextRecognition(text="ABC123", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
            TextRecognition(text="@#$%", line=1, det_score=0.9, rec_score=0.8, polygon=[]),  # Only special chars - becomes ""
            TextRecognition(text="X", line=2, det_score=0.9, rec_score=0.8, polygon=[]),  # Short but valid chars
        ]
        cleaned = normalizer.clean_results(results)
        # ABC123 and X pass (valid chars), @#$% filtered (becomes empty after normalize)
        assert len(cleaned) == 2
        assert cleaned[0].text == "ABC123"
        assert cleaned[1].text == "X"


class TestMultiLineConcatenator:
    """Test MultiLineConcatenator."""

    def test_concatenate_single_line(self):
        concat = MultiLineConcatenator()
        results = [
            TextRecognition(text="ABC123", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
        ]
        assert concat.concatenate(results) == "ABC123"

    def test_concatenate_multi_line(self):
        concat = MultiLineConcatenator()
        results = [
            TextRecognition(text="29A", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
            TextRecognition(text="12345", line=1, det_score=0.9, rec_score=0.8, polygon=[]),
        ]
        assert concat.concatenate(results) == "29A_12345"

    def test_concatenate_three_lines(self):
        concat = MultiLineConcatenator()
        results = [
            TextRecognition(text="A", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
            TextRecognition(text="B", line=1, det_score=0.9, rec_score=0.8, polygon=[]),
            TextRecognition(text="C", line=2, det_score=0.9, rec_score=0.8, polygon=[]),
        ]
        assert concat.concatenate(results) == "A_B_C"

    def test_concatenate_raw(self):
        concat = MultiLineConcatenator()
        assert concat.concatenate_raw(["ABC", "123"]) == "ABC_123"


class TestLPRPostProcessor:
    """Test LPRPostProcessor."""

    def test_process_single_line(self):
        proc = LPRPostProcessor()
        results = [
            TextRecognition(text="ABC123", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
        ]
        plate = proc.process(results)
        assert plate == "ABC123"

    def test_process_multi_line(self):
        proc = LPRPostProcessor()
        results = [
            TextRecognition(text="29A", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
            TextRecognition(text="12345", line=1, det_score=0.9, rec_score=0.8, polygon=[]),
        ]
        plate = proc.process(results)
        assert plate == "29A_12345"

    def test_process_empty(self):
        proc = LPRPostProcessor()
        assert proc.process([]) is None

    def test_process_invalid(self):
        proc = LPRPostProcessor()
        results = [
            TextRecognition(text="!!!", line=0, det_score=0.9, rec_score=0.8, polygon=[]),
        ]
        assert proc.process(results) is None
