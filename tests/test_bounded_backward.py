from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.bounded_backward import (
    GradientWorkingSetExceededError,
    run_bounded_backward,
)
from microcolossus.bounded_forward import WorkingSetExceededError
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.storage import VersionedTensorStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="bounded-backward-micro",
        output_dir=str(tmp_path / "unused"),
        model=ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=1,
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
            seed=11,
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


def test_bounded_backward_matches_resident_gradients_exactly(tmp_path: Path) -> None:
    result = run_bounded_backward(
        _config(tmp_path),
        parameter_store_path=tmp_path / "parameters",
        oracle_gradient_store_path=tmp_path / "oracle-gradients",
        gradient_store_path=tmp_path / "gradients",
        output_path=tmp_path / "result.json",
        device_override="cpu",
    )

    assert result.loss_absolute_difference == 0.0
    assert result.oracle_store_norm_absolute_difference == 0.0
    assert result.gradient_norm_absolute_difference == 0.0
    assert result.resident_vs_bounded_gradients.exact_bytes
    assert result.resident_vs_bounded_gradients.maximum_absolute_difference == 0.0
    assert result.backward_group_order == ("final-head", "block-0", "embedding")
    assert result.parameter_manifest_unchanged
    assert result.gradient_tensor_count == 12
    assert len(result.gradient_versions) == 12
    assert result.tied_gradient_accumulation_count == 2
    assert result.tied_gradient_version == 1
    assert result.retained_cpu_activations
    assert result.retained_cpu_activation_bytes > 0
    assert result.resident_oracle_released_before_bounded
    assert result.bootstrap_payloads_released_before_bounded
    assert result.full_gradient_state_materialized_for_validation
    assert result.future_clip_coefficient <= 1.0

    parameter_store = VersionedTensorStore.open(tmp_path / "parameters")
    oracle_store = VersionedTensorStore.open(tmp_path / "oracle-gradients")
    gradient_store = VersionedTensorStore.open(tmp_path / "gradients")
    assert parameter_store.verify().tensor_count == 12
    assert oracle_store.verify().tensor_count == 12
    assert gradient_store.verify().tensor_count == 12
    assert not oracle_store.recover().incomplete_transactions
    assert not gradient_store.recover().incomplete_transactions


def test_bounded_backward_is_deterministic_on_cpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_bounded_backward(
        config,
        parameter_store_path=tmp_path / "parameters-first",
        oracle_gradient_store_path=tmp_path / "oracle-gradients-first",
        gradient_store_path=tmp_path / "gradients-first",
        device_override="cpu",
    )
    second = run_bounded_backward(
        config,
        parameter_store_path=tmp_path / "parameters-second",
        oracle_gradient_store_path=tmp_path / "oracle-gradients-second",
        gradient_store_path=tmp_path / "gradients-second",
        device_override="cpu",
    )

    assert first.batch_checksum == second.batch_checksum
    assert first.resident_loss == second.resident_loss
    assert first.bounded_loss == second.bounded_loss
    assert first.resident_gradient_norm == second.resident_gradient_norm
    assert first.oracle_store_gradient_norm == second.oracle_store_gradient_norm
    assert first.bounded_gradient_norm == second.bounded_gradient_norm
    assert first.resident_vs_bounded_gradients == second.resident_vs_bounded_gradients
    assert first.backward_group_order == second.backward_group_order


def test_parameter_working_set_budget_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(WorkingSetExceededError):
        run_bounded_backward(
            _config(tmp_path),
            parameter_store_path=tmp_path / "parameters",
            oracle_gradient_store_path=tmp_path / "oracle-gradients",
            gradient_store_path=tmp_path / "gradients",
            device_override="cpu",
            parameter_working_set_bytes=1,
        )


def test_gradient_working_set_budget_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(GradientWorkingSetExceededError):
        run_bounded_backward(
            _config(tmp_path),
            parameter_store_path=tmp_path / "parameters",
            oracle_gradient_store_path=tmp_path / "oracle-gradients",
            gradient_store_path=tmp_path / "gradients",
            device_override="cpu",
            gradient_working_set_bytes=1,
        )
