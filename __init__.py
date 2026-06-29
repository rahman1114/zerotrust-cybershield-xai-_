"""ZeroTrust CyberShield XAI: an open-source explainable AI framework for
adaptive cyber defense.

A lightweight, privacy-preserving toolkit that detects anomalies in network
telemetry, *explains* why each alert fired, and *adapts* its detection posture
from analyst feedback — supporting transparent, auditable, false-positive-aware,
Zero-Trust-aligned security monitoring.
"""

from .adaptive import AdaptiveDefender, FeedbackStats
from .data import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TrafficConfig,
    generate_traffic,
    train_test_split_traffic,
)
from .detector import AnomalyDetector
from .explainer import AnomalyExplainer, Explanation, FeatureContribution
from .pipeline import EvaluationReport, evaluate, run_demo

__version__ = "0.1.0"

__all__ = [
    "FEATURE_COLUMNS",
    "LABEL_COLUMN",
    "TrafficConfig",
    "generate_traffic",
    "train_test_split_traffic",
    "AnomalyDetector",
    "AnomalyExplainer",
    "Explanation",
    "FeatureContribution",
    "EvaluationReport",
    "evaluate",
    "run_demo",
    "AdaptiveDefender",
    "FeedbackStats",
    "__version__",
]
