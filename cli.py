"""Command-line interface for ZeroTrust CyberShield XAI.

Usage examples
--------------
    cybershield demo
    cybershield adapt
    cybershield train --input flows.csv --model model.joblib
    cybershield detect --model model.joblib --input flows.csv --explain
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .adaptive import AdaptiveDefender
from .data import TrafficConfig, generate_traffic, train_test_split_traffic
from .detector import AnomalyDetector
from .explainer import AnomalyExplainer
from .pipeline import evaluate, run_demo


def _cmd_demo(args: argparse.Namespace) -> int:
    detector, report, explanations = run_demo(top_alerts=args.top)
    print(report.to_text())
    print("\nTop alert explanations:\n")
    for exp in explanations:
        print(exp.to_text(k=args.k))
        print()
    return 0


def _cmd_adapt(args: argparse.Namespace) -> int:
    """Demonstrate Zero-Trust-style adaptive defense from analyst feedback."""
    data = generate_traffic(TrafficConfig())
    train, test = train_test_split_traffic(data)
    # Split the test set into a "review" batch (feedback) and a holdout.
    review, holdout = train_test_split_traffic(test, test_frac=0.5)

    detector = AnomalyDetector().fit(train)
    before = evaluate(detector, holdout)
    print("Before adaptation:")
    print(before.to_text())

    defender = AdaptiveDefender(detector)
    defender.apply_feedback(review, truth_column="is_attack")

    after = evaluate(detector, holdout)
    print("\nAfter adapting to analyst feedback:")
    print(after.to_text())
    print("\n" + defender.stats.to_text())
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    data = pd.read_csv(args.input)
    detector = AnomalyDetector(contamination=args.contamination).fit(data)
    detector.save(args.model)
    print(f"Model trained on {len(data)} records and saved to {args.model}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    detector = AnomalyDetector.load(args.model)
    data = pd.read_csv(args.input)
    preds = detector.predict(data)
    scores = detector.anomaly_score(data)

    flagged = int(preds.sum())
    print(f"Scored {len(data)} records; flagged {flagged} as anomalous.\n")

    if args.explain:
        explainer = AnomalyExplainer(detector, background=data)
        anomalies = data[preds == 1]
        if len(anomalies) == 0:
            print("No anomalies to explain.")
        for exp in explainer.explain(anomalies):
            print(exp.to_text(k=args.k))
            print()
    else:
        out = data.copy()
        out["anomaly_score"] = scores
        out["is_anomaly"] = preds
        if args.output:
            out.to_csv(args.output, index=False)
            print(f"Results written to {args.output}")
        else:
            print(out[["anomaly_score", "is_anomaly"]].head(20).to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cybershield",
        description="ZeroTrust CyberShield XAI: explainable AI for adaptive cyber defense.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run an end-to-end demo on synthetic data.")
    p_demo.add_argument("--top", type=int, default=5, help="How many alerts to explain.")
    p_demo.add_argument("--k", type=int, default=3, help="Top features per explanation.")
    p_demo.set_defaults(func=_cmd_demo)

    p_adapt = sub.add_parser(
        "adapt", help="Demo Zero-Trust adaptive defense from analyst feedback."
    )
    p_adapt.set_defaults(func=_cmd_adapt)

    p_train = sub.add_parser("train", help="Train a detector on a CSV of flows.")
    p_train.add_argument("--input", required=True, help="CSV with feature columns.")
    p_train.add_argument("--model", required=True, help="Output path for the model.")
    p_train.add_argument("--contamination", type=float, default=0.05)
    p_train.set_defaults(func=_cmd_train)

    p_detect = sub.add_parser("detect", help="Detect anomalies in a CSV of flows.")
    p_detect.add_argument("--model", required=True, help="Path to a saved model.")
    p_detect.add_argument("--input", required=True, help="CSV with feature columns.")
    p_detect.add_argument("--output", help="Optional CSV path for scored results.")
    p_detect.add_argument("--explain", action="store_true", help="Explain each alert.")
    p_detect.add_argument("--k", type=int, default=3)
    p_detect.set_defaults(func=_cmd_detect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
