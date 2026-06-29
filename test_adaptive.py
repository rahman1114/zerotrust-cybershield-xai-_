"""Tests for the adaptive defense module."""

import pytest

from cybershield_xai import (
    AdaptiveDefender,
    AnomalyDetector,
    TrafficConfig,
    generate_traffic,
)


@pytest.fixture
def fitted_detector():
    data = generate_traffic(TrafficConfig(n_normal=400, n_attacks=40, seed=3))
    return AnomalyDetector().fit(data), data


def test_requires_fitted_detector():
    with pytest.raises(RuntimeError):
        AdaptiveDefender(AnomalyDetector())


def test_false_positive_makes_stricter(fitted_detector):
    detector, _ = fitted_detector
    defender = AdaptiveDefender(detector, step=0.05)
    start = detector.threshold
    defender.register_false_positive()
    assert detector.threshold < start
    assert defender.stats.false_positives == 1


def test_false_negative_makes_more_sensitive(fitted_detector):
    detector, _ = fitted_detector
    defender = AdaptiveDefender(detector, step=0.05)
    start = detector.threshold
    defender.register_false_negative()
    assert detector.threshold > start
    assert defender.stats.false_negatives == 1


def test_threshold_stays_within_bounds(fitted_detector):
    detector, _ = fitted_detector
    defender = AdaptiveDefender(detector, step=0.5, min_threshold=-0.1, max_threshold=0.1)
    for _ in range(20):
        defender.register_false_negative()
    assert detector.threshold <= 0.1


def test_apply_feedback_records_stats(fitted_detector):
    detector, data = fitted_detector
    defender = AdaptiveDefender(detector)
    stats = defender.apply_feedback(data, truth_column="is_attack")
    assert stats.total_feedback() > 0
    assert stats.to_text()


def test_apply_feedback_missing_column(fitted_detector):
    detector, data = fitted_detector
    defender = AdaptiveDefender(detector)
    with pytest.raises(ValueError):
        defender.apply_feedback(data.drop(columns=["is_attack"]), truth_column="is_attack")
