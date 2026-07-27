from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.bounded_optimizer import (
    OptimizerWorkingSetExceededError,
    run_bounded_optimizer_step,
)
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.step_bundle import StepBundleStore
from microcolossus.storage import VersionedTensorStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="bounded-optimizer-micro",
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
            nvme_gib=0.05,
            ssd_write_budget_tb=1.0,
            memory_architecture="unified",
            system_memory_gib=1.0,
        ),
    )


def test_bounded_optimizer_matches_resident_state_exactly(tmp_path: Path) -> None:
    result = run_bounded_optimizer_step(
        _config(tmp_path),
        bundle_store_path=tmp_path / "bundle",
        output_path=tmp_path / "result.json",
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    assert result.loss_absolute_difference == 0.0
    assert result.gradient_norm_absolute_difference == 0.0
    assert result.resident_vs_candidate_state.exact_bytes
    assert result.resident_vs_candidate_state.maximum_absolute_difference == 0.0
    assert result.candidate_vs_restored_state.exact_bytes
    assert result.optimizer_group_order == ("embedding", "block-0", "final-head")
    assert result.tied_parameter_update_count == 1
    assert result.initial_bundle_step == 0
    assert result.final_bundle_step == 1
    assert result.initial_bundle_remained_authoritative_until_final_publish
    assert result.final_bundle_is_authoritative
    assert result.parameter_budget_respected
    assert result.gradient_budget_respected
    assert result.optimizer_budget_respected
    assert all(version == 1 for _, version in result.candidate_tensor_versions)

    bundle = StepBundleStore.open(tmp_path / "bundle")
    assert bundle.verify().committed_step == 1
    assert bundle.recover().current_bundle_id == result.final_bundle_id
    assert VersionedTensorStore.open(result.candidate_parameter_store_path).verify().tensor_count == 12
    assert VersionedTensorStore.open(result.candidate_optimizer_store_path).verify().tensor_count == 37


def test_bounded_optimizer_is_deterministic_on_cpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_bounded_optimizer_step(
        config,
        bundle_store_path=tmp_path / "first",
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    second = run_bounded_optimizer_step(
        config,
        bundle_store_path=tmp_path / "second",
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    assert first.batch_checksum == second.batch_checksum
    assert first.resident_loss == second.resident_loss
    assert first.bounded_loss == second.bounded_loss
    assert first.clipping_coefficient == second.clipping_coefficient
    assert first.resident_vs_candidate_state == second.resident_vs_candidate_state
    assert first.optimizer_group_order == second.optimizer_group_order
    assert first.candidate_tensor_versions == second.candidate_tensor_versions


def test_optimizer_working_set_budget_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(OptimizerWorkingSetExceededError):
        run_bounded_optimizer_step(
            _config(tmp_path),
            bundle_store_path=tmp_path / "bundle",
            device_override="cpu",
            optimizer_working_set_bytes=1,
        )
