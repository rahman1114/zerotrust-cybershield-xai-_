"""Tests for the explainer and evaluation pipeline."""

import pytest

from cybershield_xai import (
    AnomalyDetector,
    AnomalyExplainer,
    TrafficConfig,
    evaluate,
    generate_traffic,
    run_demo,
)


@pytest.fixture
def traffic():
    return generate_traffic(TrafficConfig(n_normal=400, n_attacks=40, seed=7))


def test_explanations_cover_all_features(traffic):
    detector = AnomalyDetector().fit(traffic)
    explainer = AnomalyExplainer(detector, background=traffic, max_background=50)
    sample = traffic.head(3)
    explanations = explainer.explain(sample)

    assert len(explanations) == 3
    for exp in explanations:
        assert len(exp.contributions) == len(detector.feature_columns)
        assert exp.to_text()  # renders without error


def test_explanation_top_k(traffic):
    detector = AnomalyDetector().fit(traffic)
    explainer = AnomalyExplainer(detector, background=traffic, max_background=50)
    exp = explainer.explain(traffic.head(1))[0]
    assert len(exp.top(2)) == 2
    # Top contributions should be sorted by absolute impact.
    impacts = [abs(c.contribution) for c in exp.top(3)]
    assert impacts == sorted(impacts, reverse=True)


def test_evaluation_report(traffic):
    detector = AnomalyDetector().fit(traffic)
    report = evaluate(detector, traffic)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.false_positive_rate <= 1.0
    assert report.n_total == len(traffic)


def test_run_demo_smoke():
    detector, report, explanations = run_demo(
        config=TrafficConfig(n_normal=300, n_attacks=30), top_alerts=3
    )
    assert report.n_total > 0
    assert len(explanations) == 3
