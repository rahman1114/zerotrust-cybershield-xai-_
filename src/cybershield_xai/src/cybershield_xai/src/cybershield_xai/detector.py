"""Anomaly detection for cybersecurity monitoring.

The detector wraps scikit-learn's :class:`~sklearn.ensemble.IsolationForest`
in a small, opinionated API tailored to security telemetry:

* unsupervised training (no labels required),
* standardized features for stable behaviour,
* a tunable decision threshold so analysts can trade recall against the
  false-positive rate — a central concern in your proposed endeavor.

Isolation Forest is a good default for tabular flow data: it is fast, scales
to large volumes, and produces per-record anomaly scores that the explainer
module can then attribute back to individual features.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS


class AnomalyDetector:
    """Unsupervised anomaly detector for network-flow records."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._threshold: float | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, data: pd.DataFrame) -> "AnomalyDetector":
        """Fit the scaler and isolation forest on (unlabelled) traffic."""
        features = self._select_features(data)
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)

        # Default threshold: the contamination-th percentile of scores.
        scores = self._raw_scores(scaled)
        self._threshold = float(np.quantile(scores, self.contamination))
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def anomaly_score(self, data: pd.DataFrame) -> np.ndarray:
        """Return an anomaly score per row. Higher = more anomalous."""
        self._check_fitted()
        scaled = self.scaler.transform(self._select_features(data))
        # Invert so that larger numbers mean "more anomalous" (more intuitive).
        return -self._raw_scores(scaled)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return 1 for anomalies (likely attacks), 0 for normal traffic."""
        self._check_fitted()
        scaled = self.scaler.transform(self._select_features(data))
        scores = self._raw_scores(scaled)
        return (scores < self._threshold).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Manually override the decision threshold (raw-score space).

        Lower thresholds flag fewer records (fewer false positives, lower
        recall); higher thresholds flag more. Exposed so SOC teams can tune
        precision/recall to their alert budget.
        """
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float | None:
        return self._threshold

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        self._check_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyDetector":
        return joblib.load(path)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _raw_scores(self, scaled: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal in sklearn's convention.
        return self.model.decision_function(scaled)

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        return data[self.feature_columns]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detector is not fitted. Call .fit(data) first.")
