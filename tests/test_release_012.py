from pathlib import Path

import microcolossus

from microcolossus.config import load_experiment_config
from microcolossus.model import DecoderOnlyTransformer
from microcolossus.training_checkpoint import (
    ACTIVATION_RECOMPUTE_RUNTIME_VERSION,
    BOUNDED_TRAINING_SCHEMA_VERSION,
    HYBRID_ACTIVATION_RUNTIME_VERSION,
)


def test_release_012_contract() -> None:
    micro = load_experiment_config(Path("examples/real-text-micro-recompute.yaml"))
    tiny = load_experiment_config(Path("examples/tiny-mps-recompute.yaml"))
    small = load_experiment_config(Path("examples/real-text-small-recompute.yaml"))
    hybrid_micro = load_experiment_config(Path("examples/real-text-micro-hybrid.yaml"))
    hybrid_tiny = load_experiment_config(Path("examples/tiny-mps-hybrid.yaml"))
    hybrid_small = load_experiment_config(Path("examples/real-text-small-hybrid.yaml"))

    assert microcolossus.__version__ == "0.13.0"
    assert BOUNDED_TRAINING_SCHEMA_VERSION == "microcolossus.bounded-training.v3"
    assert ACTIVATION_RECOMPUTE_RUNTIME_VERSION == "0.12.0"
    assert HYBRID_ACTIVATION_RUNTIME_VERSION == "0.13.0"
    assert micro.training.activation_policy == "recompute"
    assert tiny.training.activation_policy == "recompute"
    assert small.training.activation_policy == "recompute"
    assert hybrid_micro.training.activation_policy == "hybrid"
    assert hybrid_tiny.training.activation_policy == "hybrid"
    assert hybrid_small.training.activation_policy == "hybrid"
    assert DecoderOnlyTransformer(micro.model).parameter_count == 18_624
    assert DecoderOnlyTransformer(tiny.model).parameter_count == 443_648
    assert DecoderOnlyTransformer(small.model).parameter_count == 1_846_656
    assert DecoderOnlyTransformer(hybrid_micro.model).parameter_count == 18_624
    assert DecoderOnlyTransformer(hybrid_tiny.model).parameter_count == 443_648
    assert DecoderOnlyTransformer(hybrid_small.model).parameter_count == 1_846_656
