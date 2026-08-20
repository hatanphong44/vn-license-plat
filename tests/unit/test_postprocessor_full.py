"""Unit tests for pipeline postprocessor - additional coverage."""


from src.domain.models import LPRResult, TextRecognition
from src.pipeline.postprocessor import (
    LPRPostProcessor,
    MultiLineConcatenator,
    TextNormalizer,
)


class TestTextNormalizerEdgeCases:
    """Test TextNormalizer edge cases for coverage."""

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("")
        assert result == ""

    def test_normalize_with_spaces(self):
        """Test normalizing string with spaces."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("  ABC  123  ")
        assert result == "ABC123"

    def test_normalize_with_special_chars(self):
        """Test normalizing string with special characters."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("ABC-123.456")
        assert result == "ABC123456"

    def test_normalize_with_underscores(self):
        """Test normalizing string preserves underscores."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("ABC_123")
        assert result == "ABC_123"

    def test_normalize_non_string(self):
        """Test normalizing non-string input."""
        normalizer = TextNormalizer()
        result = normalizer.normalize(12345)
        assert result == "12345"

    def test_is_valid_empty_string(self):
        """Test validation of empty string."""
        normalizer = TextNormalizer()
        assert normalizer.is_valid("") is False

    def test_is_valid_underscore_only(self):
        """Test validation of underscore-only text."""
        normalizer = TextNormalizer()
        # Undercores are removed for validation, so this becomes empty
        assert normalizer.is_valid("_") is False
        assert normalizer.is_valid("__") is False

    def test_is_valid_too_short(self):
        """Test validation of too short text."""
        normalizer = TextNormalizer(min_length=4)
        assert normalizer.is_valid("ABC") is False

    def test_is_valid_too_long(self):
        """Test validation of too long text."""
        normalizer = TextNormalizer(max_length=10)
        assert normalizer.is_valid("ABCDEFGHIJKL") is False

    def test_is_valid_with_invalid_chars(self):
        """Test validation of text with invalid characters."""
        normalizer = TextNormalizer()
        assert normalizer.is_valid("ABC-123") is False  # contains hyphen
        assert normalizer.is_valid("ABC 123") is False  # contains space
        assert normalizer.is_valid("ABC_123") is True   # underscore OK

    def test_is_valid_with_leading_trailing_underscores(self):
        """Test validation with leading/trailing underscores."""
        normalizer = TextNormalizer()
        # "ABC" after removing underscores is 3 chars, but min is 4
        assert normalizer.is_valid("_ABC") is False

    def test_clean_results_empty_list(self):
        """Test cleaning empty results."""
        normalizer = TextNormalizer()
        result = normalizer.clean_results([])
        assert result == []

    def test_clean_results_all_invalid(self):
        """Test cleaning results where all are invalid (too short)."""
        normalizer = TextNormalizer()
        text1 = TextRecognition(text="A", line=0, det_score=0.9, rec_score=0.9, polygon=[])
        text2 = TextRecognition(text="B", line=1, det_score=0.9, rec_score=0.9, polygon=[])
        result = normalizer.clean_results([text1, text2])
        # Both "A" and "B" are single chars which fail the min_length=4 validation
        # But they pass the regex validation before length check
        # The clean_results passes them through since they match VALID_CHARS
        # The length check only happens during is_valid check
        # So these results pass clean_results but would fail later
        assert len(result) == 2  # They pass the initial regex check

    def test_clean_results_updates_text(self):
        """Test that clean_results updates normalized text."""
        normalizer = TextNormalizer()
        text = TextRecognition(text="abc 123", line=0, det_score=0.9, rec_score=0.9, polygon=[])
        result = normalizer.clean_results([text])
        assert len(result) == 1
        assert result[0].text == "ABC123"


class TestMultiLineConcatenatorEdgeCases:
    """Test MultiLineConcatenator edge cases for coverage."""

    def test_concatenate_empty_list(self):
        """Test concatenating empty list."""
        concat = MultiLineConcatenator()
        result = concat.concatenate([])
        assert result == ""

    def test_concatenate_single_result(self):
        """Test concatenating single result."""
        concat = MultiLineConcatenator()
        text = TextRecognition(text="ABC123", line=0, det_score=0.9, rec_score=0.9, polygon=[])
        result = concat.concatenate([text])
        assert result == "ABC123"

    def test_concatenate_sorted_by_line(self):
        """Test concatenation sorts by line number."""
        concat = MultiLineConcatenator()
        text1 = TextRecognition(text="ABC", line=2, det_score=0.9, rec_score=0.9, polygon=[])
        text2 = TextRecognition(text="123", line=1, det_score=0.9, rec_score=0.9, polygon=[])
        result = concat.concatenate([text1, text2])
        # Should be sorted: line 1 first, then line 2
        assert result == "123_ABC"

    def test_concatenate_raw_empty_list(self):
        """Test raw concatenation with empty list."""
        concat = MultiLineConcatenator()
        result = concat.concatenate_raw([])
        assert result == ""

    def test_concatenate_raw_single(self):
        """Test raw concatenation with single item."""
        concat = MultiLineConcatenator()
        result = concat.concatenate_raw(["ABC123"])
        assert result == "ABC123"


class TestLPRPostProcessorEdgeCases:
    """Test LPRPostProcessor edge cases for coverage."""

    def test_process_empty_results(self):
        """Test processing empty OCR results."""
        processor = LPRPostProcessor()
        result = processor.process([])
        assert result is None

    def test_process_all_invalid_results(self):
        """Test processing all invalid OCR results."""
        processor = LPRPostProcessor()
        text1 = TextRecognition(text="A", line=0, det_score=0.9, rec_score=0.9, polygon=[])
        text2 = TextRecognition(text="B", line=1, det_score=0.9, rec_score=0.9, polygon=[])
        result = processor.process([text1, text2])
        assert result is None

    def test_process_result_with_empty_ocr(self):
        """Test processing LPRResult with empty OCR results."""
        processor = LPRPostProcessor()
        result = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[],
        )
        plate_text = processor.process_result(result)
        assert plate_text is None

    def test_process_result_with_invalid_ocr(self):
        """Test processing LPRResult with invalid OCR results."""
        processor = LPRPostProcessor()
        text = TextRecognition(text="A", line=0, det_score=0.9, rec_score=0.9, polygon=[])
        result = LPRResult(
            plate_index=0,
            plate="ABC123",
            plate_normalized="ABC123",
            box=[10, 20, 100, 80],
            yolo_score=0.9,
            class_name="plate",
            ocr_results=[text],
        )
        plate_text = processor.process_result(result)
        assert plate_text is None

    def test_format_for_output_normalized(self):
        """Test formatting normalized plate text."""
        processor = LPRPostProcessor()
        output = processor.format_for_output("ABC123", normalized=True)
        assert output["plate"] == "ABC123"
        assert output["plate_normalized"] == "ABC123"
        assert output["valid"] is True

    def test_format_for_output_not_normalized(self):
        """Test formatting non-normalized plate text."""
        processor = LPRPostProcessor()
        output = processor.format_for_output("abc123", normalized=False)
        assert output["plate"] == "abc123"
        # When normalized=False, it uses normalizer.normalize() which uppercases
        assert output["plate_normalized"] == "ABC123"
        # is_valid is called on original text (lowercase), so it fails
        assert output["valid"] is False

    def test_format_for_output_invalid(self):
        """Test formatting invalid plate text."""
        processor = LPRPostProcessor()
        output = processor.format_for_output("A", normalized=True)
        assert output["valid"] is False
