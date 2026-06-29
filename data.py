"""Synthetic network-traffic data generation.

This module produces realistic-looking network flow records for demos, tests,
and benchmarking. Using synthetic data keeps the project privacy-preserving:
no real enterprise or patient data is ever required to run ZeroTrust CyberShield XAI.

The feature schema loosely mirrors common NetFlow / IDS datasets (e.g. NSL-KDD,
CICIDS) so that models trained here transfer conceptually to real telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns the rest of the package expects. Keeping this in one place means the
# detector, explainer, and tests all agree on the schema.
FEATURE_COLUMNS = [
    "duration",            # connection length in seconds
    "src_bytes",           # bytes sent from source to destination
    "dst_bytes",           # bytes sent from destination to source
    "packet_count",        # total packets in the flow
    "failed_logins",       # failed authentication attempts on the flow
    "unique_ports",        # distinct destination ports touched
    "bytes_per_packet",    # derived ratio, useful for exfiltration signals
    "night_activity",      # 1 if the flow happened during off-hours, else 0
]

LABEL_COLUMN = "is_attack"


@dataclass
class TrafficConfig:
    """Tunable knobs for the synthetic generator."""

    n_normal: int = 2000
    n_attacks: int = 100
    seed: int = 42


def _generate_normal(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Benign traffic: short flows, balanced byte counts, few failed logins."""
    duration = rng.exponential(scale=2.0, size=n)
    src_bytes = rng.lognormal(mean=6.0, sigma=1.0, size=n)
    dst_bytes = rng.lognormal(mean=6.2, sigma=1.0, size=n)
    packet_count = rng.poisson(lam=12, size=n) + 1
    failed_logins = rng.binomial(n=2, p=0.02, size=n)
    unique_ports = rng.poisson(lam=1.5, size=n) + 1
    night_activity = rng.binomial(n=1, p=0.15, size=n)

    return _assemble(
        duration, src_bytes, dst_bytes, packet_count,
        failed_logins, unique_ports, night_activity, is_attack=0,
    )


def _generate_attacks(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Malicious traffic: long flows, lopsided bytes (exfiltration / scans),
    brute-force login attempts, and port-sweeping behaviour at odd hours."""
    duration = rng.exponential(scale=20.0, size=n)
    src_bytes = rng.lognormal(mean=9.5, sigma=1.5, size=n)
    dst_bytes = rng.lognormal(mean=4.0, sigma=1.2, size=n)
    packet_count = rng.poisson(lam=120, size=n) + 1
    failed_logins = rng.binomial(n=20, p=0.4, size=n)
    unique_ports = rng.poisson(lam=25, size=n) + 1
    night_activity = rng.binomial(n=1, p=0.7, size=n)

    return _assemble(
        duration, src_bytes, dst_bytes, packet_count,
        failed_logins, unique_ports, night_activity, is_attack=1,
    )


def _assemble(duration, src_bytes, dst_bytes, packet_count,
              failed_logins, unique_ports, night_activity, is_attack) -> pd.DataFrame:
    bytes_per_packet = (src_bytes + dst_bytes) / packet_count
    frame = pd.DataFrame({
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "packet_count": packet_count,
        "failed_logins": failed_logins,
        "unique_ports": unique_ports,
        "bytes_per_packet": bytes_per_packet,
        "night_activity": night_activity,
        LABEL_COLUMN: is_attack,
    })
    return frame


def generate_traffic(config: TrafficConfig | None = None) -> pd.DataFrame:
    """Generate a labelled, shuffled dataset of synthetic network flows.

    Returns a DataFrame with ``FEATURE_COLUMNS`` plus an ``is_attack`` label.
    The label is intended for *evaluation only* — the anomaly detector itself
    trains unsupervised, mirroring how real SOC teams rarely have clean labels.
    """
    config = config or TrafficConfig()
    rng = np.random.default_rng(config.seed)

    normal = _generate_normal(config.n_normal, rng)
    attacks = _generate_attacks(config.n_attacks, rng)

    data = pd.concat([normal, attacks], ignore_index=True)
    data = data.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    return data


def train_test_split_traffic(
    data: pd.DataFrame, test_frac: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple reproducible split helper."""
    shuffled = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - test_frac))
    return shuffled.iloc[:cut].copy(), shuffled.iloc[cut:].copy()
