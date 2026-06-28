# ZeroTrust CyberShield XAI

**An open-source explainable AI framework for adaptive cyber defense.**

ZeroTrust CyberShield XAI detects anomalies in network telemetry, **explains why
each alert fired**, and **adapts its detection posture from analyst feedback** —
bringing Zero Trust's "continuously verify and adapt" principle to AI-driven
security monitoring.

It is built around four goals that matter in regulated enterprise environments
(healthcare, finance, critical infrastructure):

- **Better detection** — unsupervised anomaly detection that needs no labelled attack data.
- **Fewer false positives** — a tunable threshold so teams match alerts to their analyst budget.
- **Transparency & auditability** — every alert ships with a plain-language, SHAP-based rationale showing exactly which features drove the decision.
- **Adaptive defense** — the system learns from analyst verdicts and continuously adjusts its decision boundary, never assuming a fixed level of trust.

It is **privacy-preserving by design**: the toolkit includes a synthetic traffic
generator, so you can develop, test, and benchmark without ever touching real or
sensitive data.

> Status: early-stage (v0.1.0, alpha). Contributions are very welcome — see
> [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Why this exists

Most security anomaly detectors output a score and stop there, leaving analysts
to reverse-engineer *why* something was flagged — and they stay frozen at
training time, never adapting to the live environment. ZeroTrust CyberShield XAI
pairs detection with explanation **and** continuous adaptation, so every alert is
actionable, defensible, and tuned to the threats actually being seen.

---

## Installation

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone https://github.com/your-username/zerotrust-cybershield-xai.git
cd zerotrust-cybershield-xai

# (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Or, for development (tests + linting):
pip install -e ".[dev]"
```

---

## Quick start

### 1. Run the detection + explanation demo

```bash
cybershield demo
```

Generates synthetic traffic, trains a detector, reports accuracy and the
false-positive rate, then explains the top alerts:

```
Evaluation report
  records evaluated : 630
  flagged as anomaly: 35
  accuracy          : 0.995
  precision         : 0.914
  recall            : 1.000
  false-positive rate: 0.005

Top alert explanations:

Record #1827: ANOMALY (score=0.154)
Top drivers:
  - src_bytes=253126.26 raises the anomaly score (impact +0.093)
  - failed_logins=10.00 raises the anomaly score (impact +0.070)
  - bytes_per_packet=2365.95 raises the anomaly score (impact +0.064)
```

### 2. Run the adaptive defense demo

```bash
cybershield adapt
```

Shows the detector adapting its threshold after analyst feedback on a batch of
reviewed alerts (Zero Trust continuous verification in action):

```
Adaptive feedback summary
  confirmed attacks (TP)    : 16
  false alarms (FP)         : 2
  missed attacks (FN)       : 0
  threshold adjustments     : 3
```

### 3. Use it in Python

```python
from cybershield_xai import (
    generate_traffic, train_test_split_traffic,
    AnomalyDetector, AnomalyExplainer, AdaptiveDefender, evaluate,
)

data = generate_traffic()
train, test = train_test_split_traffic(data)

# Detect
detector = AnomalyDetector(contamination=0.05).fit(train)
print(evaluate(detector, test).to_text())

# Explain
explainer = AnomalyExplainer(detector, background=train)
for exp in explainer.explain(test.head(5)):
    print(exp.to_text())

# Adapt — learn from analyst feedback
defender = AdaptiveDefender(detector)
defender.apply_feedback(test, truth_column="is_attack")
print(defender.stats.to_text())
```

### 4. Train & detect on your own CSV

Your CSV needs these feature columns: `duration`, `src_bytes`, `dst_bytes`,
`packet_count`, `failed_logins`, `unique_ports`, `bytes_per_packet`,
`night_activity`.

```bash
cybershield train  --input flows.csv --model model.joblib
cybershield detect --model model.joblib --input flows.csv --explain
```

---

## How it works

| Stage       | Module                        | What it does |
|-------------|-------------------------------|--------------|
| Data        | `cybershield_xai.data`        | Synthetic, privacy-safe network-flow generator. |
| Detection   | `cybershield_xai.detector`    | `IsolationForest` + feature scaling + a tunable threshold. |
| Explanation | `cybershield_xai.explainer`   | SHAP attributions → per-alert, human-readable reasons. |
| Adaptation  | `cybershield_xai.adaptive`    | Feedback-driven, Zero-Trust-style threshold adaptation. |
| Pipeline    | `cybershield_xai.pipeline`    | End-to-end orchestration + accuracy / false-positive metrics. |
| Interface   | `cybershield_xai.cli`         | `cybershield` command-line tool. |

---

## Roadmap

This project is the practical companion to a broader research endeavor in
explainable, privacy-preserving AI for cybersecurity. Planned directions:

- Autoencoder and ensemble detectors alongside Isolation Forest.
- Real dataset adapters (NSL-KDD, CICIDS) with the same feature schema.
- Federated / privacy-preserving training for distributed Zero Trust settings.
- LLM-generated natural-language alert narratives layered on the SHAP output.

See [issues](https://github.com/your-username/zerotrust-cybershield-xai/issues) for ways to help.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # run the test suite
ruff check .    # lint
```

---

## License

[MIT](LICENSE) © 2026 Md. Arifur Rahman
