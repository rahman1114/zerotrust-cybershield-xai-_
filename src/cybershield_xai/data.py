"""Synthetic, privacy-safe network traffic generation.

Provides a deterministic generator of labelled network-flow records so the
toolkit can be developed, tested, and benchmarked without ever touching real
or sensitive data. This is the privacy-preserving foundation of the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packet_count",
    "failed_logins",
    "unique_ports",
    "bytes_per_packet",
    "night_activity",
]

LABEL_COLUMN = "is_attack"


@dataclass
class TrafficConfig:
    """Configuration for synthetic traffic generation."""

    n_normal: int = 1800
    n_attacks: int = 200
    seed: int = 42


def generate_traffic(config: TrafficConfig | None = None) -> pd.DataFrame:
    """Generate a labelled, privacy-safe synthetic network-flow dataset."""
    config = config or TrafficConfig()
    rng = np.random.default_rng(config.seed)

    normal = pd.DataFrame({
        "duration": rng.normal(30, 10, config.n_normal).clip(1),
        "src_bytes": rng.normal(4000, 1200, config.n_normal).clip(50),
        "dst_bytes": rng.normal(3500, 1000, config.n_normal).clip(50),
        "packet_count": rng.normal(40, 12, config.n_normal).clip(1),
        "failed_logins": rng.poisson(0.3, config.n_normal),
        "unique_ports": rng.poisson(2, config.n_normal) + 1,
        "night_activity": rng.binomial(1, 0.15, config.n_normal),
        LABEL_COLUMN: 0,
    })

    attacks = pd.DataFrame({
        "duration": rng.normal(120, 45, config.n_attacks).clip(1),
        "src_bytes": rng.normal(250000, 80000, config.n_attacks).clip(1000),
        "dst_bytes": rng.normal(20000, 8000, config.n_attacks).clip(100),
        "packet_count": rng.normal(250, 90, config.n_attacks).clip(1),
        "failed_logins": rng.poisson(8, config.n_attacks),
        "unique_ports": rng.poisson(20, config.n_attacks) + 3,
        "night_activity": rng.binomial(1, 0.75, config.n_attacks),
        LABEL_COLUMN: 1,
    })

    data = pd.concat([normal, attacks], ignore_index=True)
    data["bytes_per_packet"] = data["src_bytes"] / data["packet_count"]
    return data.sample(frac=1, random_state=config.seed).reset_index(drop=True)


def train_test_split_traffic(
    data: pd.DataFrame, test_frac: float = 0.30, seed: int = 42
):
    """Split into train (normal only) and test (mixed) sets.

    The detector trains unsupervised on normal traffic, so the training split
    keeps only benign records; the test split keeps the natural mix for
    evaluation.
    """
    test = data.sample(frac=test_frac, random_state=seed)
    train = data.drop(test.index)
    train = train[train[LABEL_COLUMN] == 0]
    return train.reset_index(drop=True), test.reset_index(drop=True)
