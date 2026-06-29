"""Adaptive cyber defense.

A core idea in Zero Trust security is *continuous verification*: the system
never assumes a fixed level of trust and keeps adapting as new evidence
arrives. This module brings that idea to the detector's decision boundary.

:class:`AdaptiveDefender` wraps a fitted :class:`AnomalyDetector` and lets a
security analyst feed back verdicts on individual alerts ("this was a real
attack" / "this was a false alarm"). It uses that feedback to nudge the
detection threshold:

* too many confirmed **false positives** -> become *stricter* (flag less),
* a **missed attack** (false negative) -> become *more sensitive* (flag more).

This produces a defense posture that adapts to the live environment instead of
staying frozen at training time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .detector import AnomalyDetector


@dataclass
class FeedbackStats:
    """Running tally of analyst feedback."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    threshold_history: list[float] = field(default_factory=list)

    def total_feedback(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives

    def to_text(self) -> str:
        return (
            "Adaptive feedback summary\n"
            f"  confirmed attacks (TP)    : {self.true_positives}\n"
            f"  false alarms (FP)         : {self.false_positives}\n"
            f"  missed attacks (FN)       : {self.false_negatives}\n"
            f"  threshold adjustments     : {len(self.threshold_history)}"
        )


class AdaptiveDefender:
    """Wraps a detector with feedback-driven, Zero-Trust-style adaptation."""

    def __init__(
        self,
        detector: AnomalyDetector,
        step: float = 0.01,
        min_threshold: float = -1.0,
        max_threshold: float = 1.0,
    ) -> None:
        if detector.threshold is None:
            raise RuntimeError("Detector must be fitted before adaptive defense.")
        self.detector = detector
        self.step = step
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.stats = FeedbackStats(threshold_history=[detector.threshold])

    # ------------------------------------------------------------------ #
    # Feedback handlers
    # ------------------------------------------------------------------ #
    def register_false_positive(self, count: int = 1) -> None:
        """Analyst confirmed an alert was a false alarm -> become stricter."""
        self.stats.false_positives += count
        # Lower the raw-score threshold so fewer records get flagged.
        self._shift_threshold(-self.step * count)

    def register_false_negative(self, count: int = 1) -> None:
        """A real attack slipped through -> become more sensitive."""
        self.stats.false_negatives += count
        # Raise the raw-score threshold so more records get flagged.
        self._shift_threshold(self.step * count)

    def register_true_positive(self, count: int = 1) -> None:
        """Analyst confirmed a correct alert -> no change, just record it."""
        self.stats.true_positives += count

    def apply_feedback(self, data: pd.DataFrame, truth_column: str) -> FeedbackStats:
        """Batch-learn from a labelled slice of recently reviewed alerts.

        ``data`` must contain the feature columns plus a ground-truth column
        (1 = real attack, 0 = benign). The defender compares its predictions
        against the truth and adapts accordingly.
        """
        if truth_column not in data.columns:
            raise ValueError(f"'{truth_column}' column not found for feedback.")

        preds = self.detector.predict(data)
        truth = data[truth_column].to_numpy()

        for p, t in zip(preds, truth):
            if p == 1 and t == 1:
                self.register_true_positive()
            elif p == 1 and t == 0:
                self.register_false_positive()
            elif p == 0 and t == 1:
                self.register_false_negative()
        return self.stats

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _shift_threshold(self, delta: float) -> None:
        current = self.detector.threshold or 0.0
        new = max(self.min_threshold, min(self.max_threshold, current + delta))
        self.detector.set_threshold(new)
        self.stats.threshold_history.append(new)
