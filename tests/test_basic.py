from cybershield_xai import (
    generate_traffic,
    train_test_split_traffic,
    AnomalyDetector,
    evaluate,
)


def test_generate_traffic():
    data = generate_traffic(n_normal=100, n_attack=10)
    assert len(data) == 110
    assert "is_attack" in data.columns


def test_detector_runs():
    data = generate_traffic(n_normal=200, n_attack=20)
    train, test = train_test_split_traffic(data)
    detector = AnomalyDetector().fit(train)
    predictions = detector.predict(test)
    assert len(predictions) == len(test)


def test_evaluation_report():
    data = generate_traffic(n_normal=200, n_attack=20)
    train, test = train_test_split_traffic(data)
    detector = AnomalyDetector().fit(train)
    report = evaluate(detector, test)
    assert report.records == len(test)
