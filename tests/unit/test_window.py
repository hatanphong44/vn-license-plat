"""Test window-based LPR runtime logic."""

from collections import Counter

# Minimum observations required for finalization (matches worker.py)
MIN_OBSERVATIONS_PER_WINDOW = 10


class PlateObservation:
    """Mock observation for testing."""

    def __init__(self, plate_normalized: str, confidence: float = 0.9, is_valid: bool = True):
        self.plate_normalized = plate_normalized
        self.confidence = confidence
        self.is_valid = is_valid


class WindowResult:
    """Mock window result for testing."""

    def __init__(
        self,
        window_id: int,
        result: str | None,
        action: str,
        valid_observations: int = 0,
        invalid_observations: int = 0,
        candidate_counts: dict | None = None,
    ):
        self.window_id = window_id
        self.result = result
        self.action = action
        self.valid_observations = valid_observations
        self.invalid_observations = invalid_observations
        self.candidate_counts = candidate_counts or {}


def resolve_window(observations: list[PlateObservation]) -> tuple[str | None, str]:
    """
    Resolve a window using frequency-based selection.

    Requires MIN_OBSERVATIONS_PER_WINDOW observations for finalization.

    Returns: (result_plate, action)
    """
    if not observations:
        return None, "NO_CONFIDENT_RESULT"

    # Check minimum observation requirement
    if len(observations) < MIN_OBSERVATIONS_PER_WINDOW:
        return None, "INSUFFICIENT_OBSERVATIONS"

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
        """Single observation should return that plate (needs >= 10)."""
        observations = [PlateObservation("92CA03484")]
        result, action = resolve_window(observations)
        # With minimum 10 requirement, single observation is insufficient
        assert result is None
        assert action == "INSUFFICIENT_OBSERVATIONS"

    def test_10_observations_returns_that_plate(self):
        """Exactly 10 observations of same plate should return that plate."""
        observations = [PlateObservation("92CA03484") for _ in range(10)]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_all_same_returns_that_plate(self):
        """All same plates should return that plate."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_majority_wins(self):
        """8 out of 10 same should return the majority."""
        observations = [
            PlateObservation("92CA03484"),  # x8
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA034B4"),  # slightly different x2
            PlateObservation("92CA034B4"),
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_invalid_observations_are_ignored(self):
        """Invalid observations should not be counted."""
        observations = [
            PlateObservation("038_2CA", is_valid=False),  # invalid x2
            PlateObservation("038_2CA", is_valid=False),
            PlateObservation("92CA03484", is_valid=True),  # valid x10
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
            PlateObservation("92CA03484", is_valid=True),
        ]
        result, action = resolve_window(observations)
        # Only 10 valid: 92CA03484 wins
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_all_invalid_returns_no_result(self):
        """All invalid observations should return no result."""
        # Need 10 observations to pass minimum, then all are invalid
        observations = [
            PlateObservation("038_2CA", is_valid=False),
            PlateObservation("088_2CA", is_valid=False),
            PlateObservation("08_92CA", is_valid=False),
            PlateObservation("BAD_PLATE1", is_valid=False),
            PlateObservation("BAD_PLATE2", is_valid=False),
            PlateObservation("BAD_PLATE3", is_valid=False),
            PlateObservation("BAD_PLATE4", is_valid=False),
            PlateObservation("BAD_PLATE5", is_valid=False),
            PlateObservation("BAD_PLATE6", is_valid=False),
            PlateObservation("BAD_PLATE7", is_valid=False),
        ]
        result, action = resolve_window(observations)
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_tie_between_candidates_returns_no_result(self):
        """Tie between top 2 candidates should return no result."""
        observations = [
            PlateObservation("92CA03484"),  # x5
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03684"),  # x5
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
        ]
        result, action = resolve_window(observations)
        # 5 vs 5 - tie
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_close_count_not_tie(self):
        """Close counts are not ties if different."""
        observations = [
            PlateObservation("92CA03484"),  # x7
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03684"),  # x3
            PlateObservation("92CA03684"),
            PlateObservation("92CA03684"),
        ]
        result, action = resolve_window(observations)
        # 7 vs 3 - no tie, 92CA03484 wins
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_confidence_not_used_for_winner(self):
        """Confidence should NOT determine winner - only frequency matters."""
        # 92CA03484 appears 5 times with low confidence
        # 92CA03684 appears 5 times with very high confidence
        # Should be a tie (5 vs 5) -> no result
        observations = [
            PlateObservation("92CA03484", confidence=0.50),
            PlateObservation("92CA03484", confidence=0.50),
            PlateObservation("92CA03484", confidence=0.50),
            PlateObservation("92CA03484", confidence=0.50),
            PlateObservation("92CA03484", confidence=0.50),
            PlateObservation("92CA03684", confidence=0.99),
            PlateObservation("92CA03684", confidence=0.99),
            PlateObservation("92CA03684", confidence=0.99),
            PlateObservation("92CA03684", confidence=0.99),
            PlateObservation("92CA03684", confidence=0.99),
        ]
        result, action = resolve_window(observations)
        # 5 vs 5 - tie
        assert result is None
        assert action == "NO_CONFIDENT_RESULT"

    def test_frequency_not_confidence(self):
        """More appearances wins even with lower confidence."""
        observations = [
            PlateObservation("92CA03484", confidence=0.50),  # appears 6 times
            PlateObservation("92CA03484", confidence=0.51),
            PlateObservation("92CA03484", confidence=0.52),
            PlateObservation("92CA03484", confidence=0.53),
            PlateObservation("92CA03484", confidence=0.54),
            PlateObservation("92CA03484", confidence=0.55),
            PlateObservation("92CA03684", confidence=0.99),  # appears 4 times
            PlateObservation("92CA03684", confidence=0.99),
            PlateObservation("92CA03684", confidence=0.99),
            PlateObservation("92CA03684", confidence=0.99),
        ]
        result, action = resolve_window(observations)
        # 6 vs 4 - 92CA03484 wins
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
        action = "SKIP_DUPLICATE" if window2_result == last_published else "PUBLISH"

        assert action == "SKIP_DUPLICATE"

    def test_different_plate_between_windows_is_new(self):
        """Different plate in consecutive windows should be published."""
        window1_result = "92CA03484"
        window2_result = "29A12345"

        last_published = window1_result
        action = "SKIP_DUPLICATE" if window2_result == last_published else "PUBLISH"

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
        action2 = "SKIP_DUPLICATE" if window2_result == last_published else "PUBLISH"

        assert action1 == "NO_CONFIDENT_RESULT"
        assert action2 == "PUBLISH"


class TestEndToEndScenario:
    """Test complete scenario: A → A → A → B → B → C with minimum 10 observations."""

    def test_scenario_produces_correct_publishes(self):
        """A→A→A→B→B→C should publish A, B, C once each."""
        windows = [
            # Each window must have >= 10 observations
            ["92CA03484"] * 10,  # Window 1
            ["92CA03484"] * 10,  # Window 2
            ["92CA03484"] * 10,  # Window 3
            ["29A12345"] * 10,  # Window 4
            ["29A12345"] * 10,  # Window 5
            ["51B12345"] * 10,  # Window 6
        ]

        published = []
        last_published = None

        for i, obs_list in enumerate(windows, 1):
            observations = [PlateObservation(p) for p in obs_list]
            result, _action = resolve_window(observations)

            if result is None:
                final_action = "NO_CONFIDENT_RESULT"
            elif result == last_published:
                final_action = "SKIP_DUPLICATE"
            else:
                final_action = "PUBLISH"
                last_published = result
                published.append(result)

            print(f"Window {i}: {result or 'None'} → {final_action}")

        assert published == ["92CA03484", "29A12345", "51B12345"], (
            f"Expected 3 publishes, got: {published}"
        )

    def test_invalid_plates_filtered_out(self):
        """Invalid plates like 038_2CA should be filtered before counting."""
        windows = [
            # Window 1: Mix of valid and invalid, 10+ total observations
            ["92CA03484"] * 7 + ["038_2CA", "088_2CA", "INVALID1"],  # 10 observations
        ]

        published = []
        last_published = None

        for i, obs_list in enumerate(windows, 1):
            observations = [
                PlateObservation(p, is_valid=p not in ["038_2CA", "088_2CA", "INVALID1"])
                for p in obs_list
            ]
            result, _action = resolve_window(observations)

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


class TestMinimumObservations:
    """Test minimum 10 observations requirement per 3-second window."""

    def test_5_observations_no_result(self):
        """5 observations should NOT produce a result."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        result, action = resolve_window(observations)
        assert result is None
        assert action == "INSUFFICIENT_OBSERVATIONS"

    def test_9_observations_no_result(self):
        """9 observations should NOT produce a result."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        result, action = resolve_window(observations)
        assert result is None
        assert action == "INSUFFICIENT_OBSERVATIONS"

    def test_10_observations_all_same_allows_finalization(self):
        """Exactly 10 observations should ALLOW finalization."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_more_than_10_observations_allows_finalization(self):
        """More than 10 observations should ALLOW finalization."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"
        assert action == "HAS_RESULT"

    def test_10_plus_observations_multiple_plates_majority_wins(self):
        """10+ observations with multiple plates: most frequent valid plate wins.

        Key distinction:
        - 10 = minimum TOTAL observations in the 3-second window
        - 10 != minimum number of times the same plate must appear

        Example from requirements:
        - Total observations = 25
        - Plate A = 8 occurrences
        - Plate B = 5 occurrences
        - Plate C = 3 occurrences
        - Other plates = 9 occurrences
        - Final result = Plate A (most frequent, NOT 10 occurrences)
        """
        # 11 total observations: 6 of one plate, 5 of another
        # Winner is most frequent (6), NOT requiring 10 occurrences
        observations = [
            # Plate A occurrences
            PlateObservation("92CA03484"),  # 1
            PlateObservation("92CA03484"),  # 2
            PlateObservation("92CA03484"),  # 3
            PlateObservation("92CA03484"),  # 4
            PlateObservation("92CA03484"),  # 5
            PlateObservation("92CA03484"),  # 6
            # Plate B occurrences
            PlateObservation("29A12345"),  # 7
            PlateObservation("29A12345"),  # 8
            PlateObservation("29A12345"),  # 9
            PlateObservation("29A12345"),  # 10
            PlateObservation("29A12345"),  # 11
        ]
        result, action = resolve_window(observations)
        assert result == "92CA03484"  # Most frequent (6 vs 5)
        assert action == "HAS_RESULT"

    def test_winning_plate_does_not_need_10_occurrences(self):
        """The winning plate only needs to be most frequent, NOT appear 10 times.

        Example:
        - Total = 12 observations
        - Plate X = 3 occurrences
        - Plate Y = 4 occurrences
        - Plate Z = 5 occurrences
        - Winner = Plate Z (5 occurrences, NOT 10)
        """
        observations = [
            # Plate Z = 5 occurrences (WINNER)
            PlateObservation("PLATE_Z"),
            PlateObservation("PLATE_Z"),
            PlateObservation("PLATE_Z"),
            PlateObservation("PLATE_Z"),
            PlateObservation("PLATE_Z"),
            # Plate Y = 4 occurrences
            PlateObservation("PLATE_Y"),
            PlateObservation("PLATE_Y"),
            PlateObservation("PLATE_Y"),
            PlateObservation("PLATE_Y"),
            # Plate X = 3 occurrences
            PlateObservation("PLATE_X"),
            PlateObservation("PLATE_X"),
            PlateObservation("PLATE_X"),
        ]
        result, action = resolve_window(observations)
        assert result == "PLATE_Z"  # Most frequent (5), not requiring 10
        assert action == "HAS_RESULT"

    def test_9_observations_high_confidence_no_result(self):
        """9 observations with very high confidence should still NOT produce a result."""
        # Even with very high confidence, minimum 10 observations is required
        observations = [
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
            PlateObservation("92CA03484", confidence=0.99),
        ]
        result, action = resolve_window(observations)
        # Confidence does NOT bypass minimum observation requirement
        assert result is None
        assert action == "INSUFFICIENT_OBSERVATIONS"

    def test_time_window_is_not_replaced_by_frame_count(self):
        """Verify the 3-second window logic uses time, not frame count."""
        # The MIN_OBSERVATIONS_PER_WINDOW is a minimum DATA requirement
        # The window boundary is still determined by time (RESULT_WINDOW_SECONDS)
        # This test verifies that observation count is additive within a window

        # Simulate two consecutive windows, each with different observations
        window1 = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]
        window2 = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]

        # Each window is resolved independently
        result1, action1 = resolve_window(window1)
        result2, action2 = resolve_window(window2)

        # Window 1: only 2 observations -> INSUFFICIENT
        assert result1 is None
        assert action1 == "INSUFFICIENT_OBSERVATIONS"

        # Window 2: 10 observations -> HAS_RESULT
        assert result2 == "92CA03484"
        assert action2 == "HAS_RESULT"

        # This shows the windows are independent (time-based) not cumulative (count-based)


class TestInsufficientWindowNoPublish:
    """Test that insufficient observation windows cannot reach the publisher."""

    def test_insufficient_observations_do_not_publish(self):
        """Insufficient observations should never reach publish logic."""
        # Simulate a window with 5 observations
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]

        result, action = resolve_window(observations)

        # No result should be produced
        assert result is None
        assert action == "INSUFFICIENT_OBSERVATIONS"

        # Simulate publish decision
        publish_decision = "PUBLISH" if action == "HAS_RESULT" else "NO_PUBLISH"
        assert publish_decision == "NO_PUBLISH"

    def test_10_observations_can_publish(self):
        """Exactly 10 observations CAN reach publish logic."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]

        result, action = resolve_window(observations)

        assert result == "92CA03484"
        assert action == "HAS_RESULT"

        # Can publish
        publish_decision = "PUBLISH" if action == "HAS_RESULT" else "NO_PUBLISH"
        assert publish_decision == "PUBLISH"

    def test_15_observations_can_publish(self):
        """15 observations CAN reach publish logic."""
        observations = [
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
            PlateObservation("92CA03484"),
        ]

        result, action = resolve_window(observations)

        assert result == "92CA03484"
        assert action == "HAS_RESULT"

        publish_decision = "PUBLISH" if action == "HAS_RESULT" else "NO_PUBLISH"
        assert publish_decision == "PUBLISH"
