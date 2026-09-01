import pytest

from faceproof.calibration import choose_threshold


def test_calibration_separates_known_scores() -> None:
    result = choose_threshold([0.71, 0.74, 0.79], [0.05, 0.12, 0.2, 0.3, 0.41, 0.49])
    assert 0.49 < result.threshold < 0.71
    assert result.balanced_accuracy == 1.0
    assert result.false_accept_rate == 0.0
    assert result.false_reject_rate == 0.0


def test_calibration_requires_minimum_validation_set() -> None:
    with pytest.raises(ValueError, match="at least 2 positive and 5 negative"):
        choose_threshold([0.7], [0.1, 0.2])


def test_calibration_prefers_lower_false_accept_rate_on_tie() -> None:
    result = choose_threshold([0.6, 0.8], [0.1, 0.2, 0.3, 0.7, 0.9])
    assert result.false_accept_rate <= 0.4
