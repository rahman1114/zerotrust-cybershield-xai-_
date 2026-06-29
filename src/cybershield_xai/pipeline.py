"""End-to-end pipeline: data -> detect -> explain -> evaluate.

Provides a single convenience entry point that ties the modules together,
plus evaluation helpers that report the metrics your endeavor emphasises:
detection accuracy and the false-positive rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import LABEL_COLUMN, TrafficConfig, generate_traffic, train_test_split_traffic
from .detector import AnomalyDetector
from .explainer import AnomalyExplainer, Explanation


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    false_positive_rate: float
    n_flagged: int
    n_total: int

    def to_text(self) -> str:
        return (
            "Evaluation report\n"
            f"  records evaluated : {self.n_total}\n"
            f"  flagged as anomaly: {self.n_flagged}\n"
            f"  accuracy          : {self.accuracy:.3f}\n"
            f"  precision         : {self.precision:.3f}\n"
            f"  recall            : {self.recall:.3f}\n"
            f"  false-positive rate: {self.false_positive_rate:.3f}"
        )


def evaluate(detector: AnomalyDetector, test: pd.DataFrame) -> EvaluationReport:
    """Score the detector against ground-truth labels (eval only)."""
    if LABEL_COLUMN not in test.columns:
        raise ValueError(f"Test set needs a '{LABEL_COLUMN}' column for evaluation.")

    y_true = test[LABEL_COLUMN].to_numpy()
    y_pred = detector.predict(test)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    return EvaluationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        n_flagged=int(np.sum(y_pred)),
        n_total=len(y_true),
    )


def run_demo(config: TrafficConfig | None = None, top_alerts: int = 5):
    """Run a complete demo end-to-end and return the key artefacts.

    Returns a tuple of (detector, report, explanations_for_top_alerts).
    """
    data = generate_traffic(config)
    train, test = train_test_split_traffic(data)

    detector = AnomalyDetector().fit(train)
    report = evaluate(detector, test)

    explainer = AnomalyExplainer(detector, background=train)

    # Explain the highest-scoring alerts in the test set.
    scores = detector.anomaly_score(test)
    order = np.argsort(scores)[::-1][:top_alerts]
    top = test.iloc[order]
    explanations: list[Explanation] = explainer.explain(top)

    return detector, report, explanations
