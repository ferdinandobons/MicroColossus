import re

import torch

from microcolossus.config import ModelConfig
from microcolossus.model import DecoderOnlyTransformer
from microcolossus.telemetry import accelerator_memory_metrics, model_checksum


def test_model_checksum_does_not_require_tensor_numpy(monkeypatch) -> None:
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=1,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
        )
    )

    def fail_numpy(_tensor: torch.Tensor) -> None:
        raise AssertionError("model_checksum must not call Tensor.numpy")

    monkeypatch.setattr(torch.Tensor, "numpy", fail_numpy)

    first = model_checksum(model)
    second = model_checksum(model)

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_model_checksum_changes_when_model_state_changes() -> None:
    model = DecoderOnlyTransformer(ModelConfig())
    before = model_checksum(model)

    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1)

    assert model_checksum(model) != before


def test_model_checksum_supports_bfloat16_state() -> None:
    model = DecoderOnlyTransformer(ModelConfig()).to(dtype=torch.bfloat16)
    assert re.fullmatch(r"[0-9a-f]{64}", model_checksum(model))


def test_mps_memory_metrics_use_mps_apis(monkeypatch) -> None:
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 11)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 22)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 33)

    metrics = accelerator_memory_metrics(torch.device("mps"))

    assert metrics.measurement_kind == "mps-current-allocated"
    assert metrics.allocated_bytes == 11
    assert metrics.driver_allocated_bytes == 22
    assert metrics.recommended_max_bytes == 33
