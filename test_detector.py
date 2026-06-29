"""Tests for the anomaly detector."""

import numpy as np
import pytest

from cybershield_xai import AnomalyDetector, TrafficConfig, generate_traffic
from cybershield_xai.data import FEATURE_COLUMNS


@pytest.fixture
def traffic():
    return generate_traffic(TrafficConfig(n_normal=500, n_attacks=40, seed=1))


def test_detector_fits_and_predicts(traffic):
    detector = AnomalyDetector().fit(traffic)
    preds = detector.predict(traffic)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(traffic)


def test_detector_flags_some_anomalies(traffic):
    detector = AnomalyDetector().fit(traffic)
    preds = detector.predict(traffic)
    # The model should flag *something* but not everything.
    assert 0 < preds.sum() < len(traffic)


def test_anomaly_scores_are_finite(traffic):
    detector = AnomalyDetector().fit(traffic)
    scores = detector.anomaly_score(traffic)
    assert np.all(np.isfinite(scores))


def test_predict_before_fit_raises(traffic):
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.predict(traffic)


def test_missing_columns_raise():
    detector = AnomalyDetector()
    bad = generate_traffic(TrafficConfig(n_normal=10, n_attacks=2)).drop(
        columns=[FEATURE_COLUMNS[0]]
    )
    with pytest.raises(ValueError):
        detector.fit(bad)


def test_threshold_tuning_changes_alert_count(traffic):
    detector = AnomalyDetector().fit(traffic)
    base = detector.predict(traffic).sum()
    # A much lower threshold should flag fewer records.
    detector.set_threshold(detector.threshold - 0.2)
    fewer = detector.predict(traffic).sum()
    assert fewer <= base


def test_save_and_load(tmp_path, traffic):
    detector = AnomalyDetector().fit(traffic)
    path = tmp_path / "model.joblib"
    detector.save(str(path))
    loaded = AnomalyDetector.load(str(path))
    assert np.array_equal(loaded.predict(traffic), detector.predict(traffic))
