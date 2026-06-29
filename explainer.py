"""Explainability layer for anomaly alerts.

This is the heart of ZeroTrust CyberShield XAI's "explainable AI" promise. When the detector
flags a flow, an analyst needs to know *why*. This module uses SHAP
(SHapley Additive exPlanations) to attribute each anomaly score to the
individual features that drove it, producing a per-alert, human-readable
rationale that supports auditability and regulatory compliance.

Each explanation answers: "Which features made this record look anomalous,
and in which direction?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap

from .detector import AnomalyDetector


@dataclass
class FeatureContribution:
    feature: str
    value: float
    contribution: float  # signed SHAP value toward the anomaly score

    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


@dataclass
class Explanation:
    """A human-readable rationale for a single anomaly decision."""

    index: int
    anomaly_score: float
    is_anomaly: bool
    contributions: list[FeatureContribution] = field(default_factory=list)

    def top(self, k: int = 3) -> list[FeatureContribution]:
        """Return the k features with the largest absolute contribution."""
        return sorted(
            self.contributions, key=lambda c: abs(c.contribution), reverse=True
        )[:k]

    def to_text(self, k: int = 3) -> str:
        """Render a short plain-language summary suitable for an alert ticket."""
        verdict = "ANOMALY" if self.is_anomaly else "normal"
        lines = [
            f"Record #{self.index}: {verdict} (score={self.anomaly_score:.3f})",
            "Top drivers:",
        ]
        for c in self.top(k):
            lines.append(
                f"  - {c.feature}={c.value:.2f} {c.direction()} the anomaly score "
                f"(impact {c.contribution:+.3f})"
            )
        return "\n".join(lines)


class AnomalyExplainer:
    """Wraps a fitted :class:`AnomalyDetector` with SHAP explanations."""

    def __init__(self, detector: AnomalyDetector, background: pd.DataFrame,
                 max_background: int = 100) -> None:
        self.detector = detector
        self.feature_columns = detector.feature_columns

        # SHAP needs a representative background sample; cap it for speed.
        bg = background[self.feature_columns]
        if len(bg) > max_background:
            bg = bg.sample(max_background, random_state=42)
        self._bg_scaled = detector.scaler.transform(bg)

        # Explain the model in *scaled* feature space, then map back.
        self._explainer = shap.Explainer(
            self._score_fn, self._bg_scaled, feature_names=self.feature_columns
        )

    def _score_fn(self, scaled: np.ndarray) -> np.ndarray:
        """Anomaly score as a function of scaled features (higher = worse)."""
        return -self.detector.model.decision_function(scaled)

    def explain(self, data: pd.DataFrame) -> list[Explanation]:
        """Produce an :class:`Explanation` for every row in ``data``."""
        features = data[self.feature_columns]
        scaled = self.detector.scaler.transform(features)
        shap_values = self._explainer(scaled)

        scores = self.detector.anomaly_score(data)
        preds = self.detector.predict(data)

        explanations: list[Explanation] = []
        for i in range(len(features)):
            contribs = [
                FeatureContribution(
                    feature=self.feature_columns[j],
                    value=float(features.iloc[i, j]),
                    contribution=float(shap_values.values[i, j]),
                )
                for j in range(len(self.feature_columns))
            ]
            explanations.append(
                Explanation(
                    index=int(data.index[i]),
                    anomaly_score=float(scores[i]),
                    is_anomaly=bool(preds[i]),
                    contributions=contribs,
                )
            )
        return explanations
