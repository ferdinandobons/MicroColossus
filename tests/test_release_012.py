from pathlib import Path

import microcolossus

from microcolossus.config import load_experiment_config
from microcolossus.training_checkpoint import (
    ACTIVATION_RECOMPUTE_RUNTIME_VERSION,
    BOUNDED_TRAINING_SCHEMA_VERSION,
)


def test_release_012_contract() -> None:
    config = load_experiment_config(Path("examples/real-text-micro-recompute.yaml"))

    assert microcolossus.__version__ == "0.12.0"
    assert BOUNDED_TRAINING_SCHEMA_VERSION == "microcolossus.bounded-training.v3"
    assert ACTIVATION_RECOMPUTE_RUNTIME_VERSION == "0.12.0"
    assert config.training.activation_policy == "recompute"
