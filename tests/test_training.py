from copy import deepcopy

import pytest
import torch

import microcolossus.training as training_module
from microcolossus.config import ModelConfig
from microcolossus.model import DecoderOnlyTransformer
from microcolossus.training import (
    make_synthetic_lm_batch,
    resolve_device,
    run_resident_step,
    seed_everything,
)


def test_auto_prefers_mps(monkeypatch) -> None:
    monkeypatch.setattr(training_module, "mps_is_available", lambda: True)
    monkeypatch.setattr(training_module, "cuda_is_available", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_explicit_mps_reports_unavailable_build(monkeypatch) -> None:
    monkeypatch.setattr(training_module, "mps_is_available", lambda: False)
    monkeypatch.setattr(training_module, "mps_is_built", lambda: False)
    with pytest.raises(RuntimeError, match="no MPS support"):
        resolve_device("mps")


def test_resident_step_is_reproducible_on_cpu() -> None:
    seed_everything(7)
    config = ModelConfig(
        vocab_size=32,
        max_sequence_length=12,
        layers=1,
        heads=2,
        hidden_size=16,
        mlp_ratio=2,
    )
    first = DecoderOnlyTransformer(config)
    second = deepcopy(first)
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=1e-3)

    generator = torch.Generator().manual_seed(9)
    input_ids, targets = make_synthetic_lm_batch(
        batch_size=2,
        sequence_length=8,
        vocab_size=config.vocab_size,
        generator=generator,
    )
    first_metrics = run_resident_step(
        model=first,
        optimizer=first_optimizer,
        input_ids=input_ids,
        targets=targets,
        device=torch.device("cpu"),
        step=0,
        gradient_clip_norm=1.0,
    )
    second_metrics = run_resident_step(
        model=second,
        optimizer=second_optimizer,
        input_ids=input_ids,
        targets=targets,
        device=torch.device("cpu"),
        step=0,
        gradient_clip_norm=1.0,
    )

    assert first_metrics.loss == second_metrics.loss
    assert first_metrics.gradient_norm == second_metrics.gradient_norm
    assert first_metrics.parameter_checksum == second_metrics.parameter_checksum
    assert first_metrics.accelerator_memory_measurement == "none"
    parameter_pairs = zip(first.parameters(), second.parameters(), strict=True)
    for first_parameter, second_parameter in parameter_pairs:
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_resident_experiment_replaces_previous_step_log(tmp_path) -> None:
    from microcolossus.config import ExperimentConfig, HardwareBudget, TrainingConfig
    from microcolossus.training import run_resident_experiment

    output_dir = tmp_path / "run"
    config = ExperimentConfig(
        name="reset-log",
        output_dir=str(output_dir),
        model=ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=1,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
        ),
        training=TrainingConfig(
            steps=1,
            micro_batch_size=1,
            sequence_length=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            seed=3,
            device="cpu",
        ),
        hardware=HardwareBudget(),
    )

    run_resident_experiment(config)
    run_resident_experiment(config)

    lines = (output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
