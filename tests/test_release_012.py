from pathlib import Path

import microcolossus

from microcolossus.config import load_experiment_config
from microcolossus.model import DecoderOnlyTransformer
from microcolossus.training_checkpoint import (
    ACTIVATION_RECOMPUTE_RUNTIME_VERSION,
    BOUNDED_TRAINING_SCHEMA_VERSION,
)


def test_release_012_contract() -> None:
    micro = load_experiment_config(Path("examples/real-text-micro-recompute.yaml"))
    tiny = load_experiment_config(Path("examples/tiny-mps-recompute.yaml"))
    small = load_experiment_config(Path("examples/real-text-small-recompute.yaml"))

    assert microcolossus.__version__ == "0.12.0"
    assert BOUNDED_TRAINING_SCHEMA_VERSION == "microcolossus.bounded-training.v3"
    assert ACTIVATION_RECOMPUTE_RUNTIME_VERSION == "0.12.0"
    assert micro.training.activation_policy == "recompute"
    assert tiny.training.activation_policy == "recompute"
    assert small.training.activation_policy == "recompute"
    assert DecoderOnlyTransformer(micro.model).parameter_count == 18_624
    assert DecoderOnlyTransformer(tiny.model).parameter_count == 443_648
    assert DecoderOnlyTransformer(small.model).parameter_count == 1_846_656
