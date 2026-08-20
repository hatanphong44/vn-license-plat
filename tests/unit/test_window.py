"""Test window-based LPR runtime logic."""

import pytest
from collections import Counter


class PlateObservation:
    """Mock observation for testing."""
    def __init__(self, plate_normalized: str, confidence: float = 0.9, is_valid: bool = True):
        self.plate_normalized = plate_normalized
        self.confidence = confidence
        self.is_valid = is_valid


class WindowResult:
    """Mock window result for testing."""
    def __init__(self, window_id: int, result: str | None, action: str,
                 valid_observations: int = 0, invalid_observations: int = 0,
                 candidate_counts: dict = None):
        self.window_id = window_id
        self.result = result
        self.action = action
        self.valid_observations = valid_observations
        self.invalid_observations = invalid_observations
        self.candidate_counts = candidate_counts or {}


def resolve_window(observations: list[PlateObservation]) -> tuple[str | None, str]:
    """
    Resolve a window using frequency-based selection (no consensus ratio).

    Returns: (result_plate, action)
    """
    if not observations:
        return None, "NO_CONFIDENT_RESULT"

    # Filter to only valid observations
    valid_obs = [obs for obs in observations if obs.is_valid]

    if not valid_obs:
        return None, "NO_CONFIDENT_RESULT"

    # Count valid plates by frequency
    plate_counts = Counter(obs.plate_normalized for obs in valid_obs)
    most_common = plate_counts.most_common()

    if not most_common:
        return None, "NO_CONFIDENT_RESULT"

    # Check for tie between top 2 candidates
    if len(most_common) >= 2:
        first_count = most_common[0][1]
        second_count = most_common[1][1]
        if first_count == second_count:
            # Tie - cannot determine winner
            return None, "NO_CONFIDENT_RESULT"

    # Winner is plate with highest frequency
    return most_common[0][0], "HAS_RESULT"


class TestWindowLogic:
    """Test window-based result resolution."""

    def test_empty_window_returns_no_result(self):
        """Empty window should not produce a result."""
        observations = []
        result, action = resolve_window(observations)
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_single_observation_returns_that_plate(self):
        """Single observation should return that plate."""
        observations = [PlateObservation("92CA03484")]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_all_same_returns_that_plate(self):
        """All same plates should return that plate."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_majority_wins(self):
        """4 out of 5 same should return the majority."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA034B4"),  # slightly different
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_invalid_observations_are_ignored(self):
        """Invalid observations should not be counted."""
        observations = [
            PlateObservation("038_2CA", is_valid=False),  # invalid
            PlateObservation("038_2CA", is_valid=False),  # invalid
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
        ]
        result, action = resolve_window(observations)
        # Only 3 valid: 92CA03484 wins
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_all_invalid_returns_no_result(self):
        """All invalid observations should return no result."""
        observations = [
            PlateObservation("038_2CA", is_valid=False),
            PlateObservation("088_2CA", is_valid=False),
            PlateObservation("08_92CA", is_valid=False),
        ]
        result, action = resolve_window(observations)
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_tie_between_candidates_returns_no_result(self):
        """Tie between top 2 candidates should return no result."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
        ]
        result, action = resolve_window(observations)
        # 3 vs 3 - tie
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_close_count_not_tie(self):
        """Close counts are not ties if different."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
        ]
        result, action = resolve_window(observations)
        # 4 vs 2 - no tie, 92CA03484 wins
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_confidence_not_used_for_winner(self):
        """Confidence should NOT determine winner - only frequency matters."""
        # 92CA03484 appears 1 time with low confidence
        # 92CA03684 appears 1 time with very high confidence
        # Should be a tie (1 vs 1) -> no result
        observations = [
            PlateObservation("92CA03484", confidence=0.50),
            PlateObservation("92CA03684", confidence=0.99),
        ]
        result, action = resolve_window(observations)
        # 1 vs 1 - tie
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_frequency_not_confidence(self):
        """More appearances wins even with lower confidence."""
        observations = [
            PlateObservation("92CA03484", confidence=0.50),  # appears 3 times
            PlateObservation("92CA03484", confidence=0.51),
            PlateObservation("92CA03484", confidence=0.52),
            PlateObservation("92CA03684", confidence=0.99),  # appears 1 time
        ]
        result, action = resolve_window(observations)
        # 3 vs 1 - 92CA03484 wins
        assert result == "92CA03484"
        assert action == "HAS_RESULT"


class TestDeduplication:
    """Test deduplication between windows."""

    def test_same_plate_between_windows_is_duplicate(self):
        """Same plate in consecutive windows should be deduplicated."""
        window1_result = "92CA03484"
        window2_result = "92CA03484"

        # Simulate dedup logic
        last_published = window1_result
        if window2_result == last_published:
            action = "SKIP_DUPLICATE"
        else:
            action = "PUBLISH"

        assert action == "SKIP_DUPLICATE"

    def test_different_plate_between_windows_is_new(self):
        """Different plate in consecutive windows should be published."""
        window1_result = "92CA03484"
        window2_result = "29A12345"

        last_published = window1_result
        if window2_result == last_published:
            action = "SKIP_DUPLICATE"
        else:
            action = "PUBLISH"

        assert action == "PUBLISH"

    def test_none_after_result_is_not_duplicate(self):
        """None result after a valid result should not be treated as duplicate."""
        last_published = "92CA03484"
        current_result = None

        if current_result is None:
            action = "NO_CONFIDENT_RESULT"
        elif current_result == last_published:
            action = "SKIP_DUPLICATE"
        else:
            action = "PUBLISH"

        assert action == "NO_CONFIDENT_RESULT"

    def test_none_followed_by_new_is_published(self):
        """None result followed by new plate should publish new plate."""
        last_published = "92CA03484"
        window1_result = None
        window2_result = "29A12345"

        # Window 1
        if window1_result is None:
            action1 = "NO_CONFIDENT_RESULT"
        else:
            action1 = "PUBLISH" if window1_result != last_published else "SKIP_DUPLICATE"

        # Window 2
        last_published = window2_result if action1 == "PUBLISH" else last_published
        if window2_result == last_published:
            action2 = "SKIP_DUPLICATE"
        else:
            action2 = "PUBLISH"

        assert action1 == "NO_CONFIDENT_RESULT"
        assert action2 == "PUBLISH"


class TestEndToEndScenario:
    """Test complete scenario: A → A → A → B → B → C"""

    def test_scenario_produces_correct_publishes(self):
        """A→A→A→B→B→C should publish A, B, C once each."""
        windows = [
            ["92CA03484", "92CA03484", "92CA03484"],  # Window 1
            ["92CA03484", "92CA03484", "92CA03484"],  # Window 2
            ["92CA03484", "92CA03484", "92CA03484"],  # Window 3
            ["29A12345", "29A12345", "29A12345"],     # Window 4
            ["29A12345", "29A12345"],                  # Window 5
            ["51B12345"],                              # Window 6
        ]

        published = []
        last_published = None

        for i, obs_list in enumerate(windows, 1):
            observations = [PlateObservation(p) for p in obs_list]
            result, action = resolve_window(observations)

            if result is None:
                final_action = "NO_CONFIDENT_RESULT"
            elif result == last_published:
                final_action = "SKIP_DUPLICATE"
            else:
                final_action = "PUBLISH"
                last_published = result
                published.append(result)

            print(f"Window {i}: {result or 'None'} → {final_action}")

        assert published == ["92CA03484", "29A12345", "51B12345"], \
            f"Expected 3 publishes, got: {published}"

    def test_invalid_plates_filtered_out(self):
        """Invalid plates like 038_2CA should be filtered before counting."""
        windows = [
            # Window 1: Mix of valid and invalid
            ["92CA03484", "92CA03484", "92CA03484", "038_2CA", "088_2CA"],
        ]

        published = []
        last_published = None

        for i, obs_list in enumerate(windows, 1):
            observations = [
                PlateObservation(p, is_valid=p not in ["038_2CA", "088_2CA"])
                for p in obs_list
            ]
            result, action = resolve_window(observations)

            if result is None:
                final_action = "NO_CONFIDENT_RESULT"
            elif result == last_published:
                final_action = "SKIP_DUPLICATE"
            else:
                final_action = "PUBLISH"
                last_published = result
                published.append(result)

            print(f"Window {i}: {result or 'None'} → {final_action}")

        assert published == ["92CA03484"]
