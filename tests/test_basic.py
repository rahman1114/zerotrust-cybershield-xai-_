from cybershield_xai import (
    AnomalyDetector,
    TrafficConfig,
    evaluate,
    generate_traffic,
    train_test_split_traffic,
)


def test_generate_traffic():
    data = generate_traffic(TrafficConfig(n_normal=100, n_attacks=10, seed=1))
    assert len(data) == 110
    assert "is_attack" in data.columns


def test_detector_runs():
    data = generate_traffic(TrafficConfig(n_normal=200, n_attacks=20, seed=2))
    train, test = train_test_split_traffic(data)
    detector = AnomalyDetector().fit(train)
    predictions = detector.predict(test)
    assert len(predictions) == len(test)


def test_evaluation_report():
    data = generate_traffic(TrafficConfig(n_normal=200, n_attacks=20, seed=3))
    train, test = train_test_split_traffic(data)
    detector = AnomalyDetector().fit(train)
    report = evaluate(detector, test)
    assert report.n_total == len(test)
    assert 0.0 <= report.accuracy <= 1.0
