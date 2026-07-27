from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcolossus.bounded_forward import (
    WorkingSetExceededError,
    build_execution_groups,
    run_bounded_forward,
)
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.storage import VersionedTensorStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="bounded-forward-micro",
        output_dir=str(tmp_path / "unused"),
        model=ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=2,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
            dropout=0.0,
        ),
        training=TrainingConfig(
            steps=1,
            micro_batch_size=1,
            sequence_length=4,
            learning_rate=1e-3,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
            seed=17,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1.0,
            process_ram_gib=1.0,
            nvme_gib=0.02,
            ssd_write_budget_tb=1.0,
            memory_architecture="unified",
            system_memory_gib=1.0,
        ),
    )


def test_bounded_forward_matches_resident_exactly_on_cpu(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = run_bounded_forward(
        _config(tmp_path),
        store_path=tmp_path / "store",
        output_path=output,
        device_override="cpu",
        parameter_working_set_bytes=1024**2,
    )

    assert json.loads(output.read_text())["schema_version"].endswith("v1")
    assert result.loss_absolute_difference == 0.0
    assert result.logits_comparison.exact_bytes
    assert result.logits_comparison.maximum_absolute_difference == 0.0
    assert all(
        item.resident_comparison.exact_bytes for item in result.execution_groups
    )
    assert result.budget_respected
    assert result.maximum_group_parameter_bytes <= result.parameter_working_set_budget_bytes
    assert result.bootstrap_model_released_before_bounded
    assert result.resident_model_released_before_bounded
    assert result.retained_activations_during_bounded
    assert result.manifest_unchanged
    assert result.repeated_tensor_reads == 1
    assert result.total_tensor_reads == result.unique_tensor_reads + 1
    assert result.execution_groups[0].name == "embedding"
    assert result.execution_groups[-1].name == "final-head"
    assert result.execution_groups[-1].tensor_names[-1] == "model.token_embedding.weight"
    store = VersionedTensorStore.open(tmp_path / "store")
    assert store.current_manifest().committed_step == 0
    store.verify()


def test_bounded_forward_is_deterministic_on_cpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_bounded_forward(
        config,
        store_path=tmp_path / "first",
        device_override="cpu",
    )
    second = run_bounded_forward(
        config,
        store_path=tmp_path / "second",
        device_override="cpu",
    )

    assert first.batch_checksum == second.batch_checksum
    assert first.resident_loss == second.resident_loss
    assert first.bounded_loss == second.bounded_loss
    assert first.resident_logits_checksum == second.resident_logits_checksum
    assert first.bounded_logits_checksum == second.bounded_logits_checksum
    assert tuple(item.output_checksum for item in first.execution_groups) == tuple(
        item.output_checksum for item in second.execution_groups
    )


def test_bounded_forward_rejects_insufficient_group_budget(tmp_path: Path) -> None:
    with pytest.raises(WorkingSetExceededError):
        run_bounded_forward(
            _config(tmp_path),
            store_path=tmp_path / "small-budget",
            device_override="cpu",
            parameter_working_set_bytes=1,
        )


def test_execution_plan_uses_one_block_per_group_and_reloads_tied_head() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_sequence_length=8,
        layers=2,
        heads=2,
        hidden_size=16,
        mlp_ratio=2,
        tie_embeddings=True,
    )
    experiment = ExperimentConfig(
        name="plan",
        output_dir="unused",
        model=config,
        training=TrainingConfig(sequence_length=4),
        hardware=HardwareBudget(),
    )
    names = {
        "model.token_embedding.weight",
        "model.position_embedding.weight",
        "model.final_norm.weight",
        "model.final_norm.bias",
    }
    for index in range(2):
        prefix = f"model.blocks.{index}"
        names.update(
            {
                f"{prefix}.attention_norm.weight",
                f"{prefix}.attention_norm.bias",
                f"{prefix}.attention.qkv.weight",
                f"{prefix}.attention.output.weight",
                f"{prefix}.mlp_norm.weight",
                f"{prefix}.mlp_norm.bias",
                f"{prefix}.mlp.input.weight",
                f"{prefix}.mlp.output.weight",
            }
        )

    groups = build_execution_groups(experiment, names)

    assert [item.name for item in groups] == [
        "embedding",
        "block-0",
        "block-1",
        "final-head",
    ]
    assert groups[-1].tensor_names[-1] == "model.token_embedding.weight"
