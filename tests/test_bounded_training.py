from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from microcolossus.bounded_optimizer import OptimizerWorkingSetExceededError
from microcolossus.bounded_training import (
    ResumeConfigurationError,
    run_bounded_training,
)
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.step_bundle import (
    BundleFailurePoint,
    BundleSimulatedCrash,
    StepBundleStore,
)
from microcolossus.storage import IntegrityError, VersionedTensorStore
from microcolossus.storage_training import compare_states


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="bounded-training-micro",
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
            steps=5,
            micro_batch_size=1,
            sequence_length=4,
            learning_rate=1e-3,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
            seed=23,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1.0,
            process_ram_gib=1.0,
            nvme_gib=0.1,
            ssd_write_budget_tb=1.0,
            memory_architecture="unified",
            system_memory_gib=1.0,
        ),
    )


def _current_state(path: Path):
    bundle = StepBundleStore.open(path)
    current = bundle.current_manifest()
    stores = [
        VersionedTensorStore.open(path / current.parameter_store.path),
        VersionedTensorStore.open(path / current.optimizer_store.path),
    ]
    return tuple(
        sorted(
            (
                store.read_tensor(record.tensor_id)
                for store in stores
                for record in store.current_manifest().tensors
            ),
            key=lambda item: item.logical_name,
        )
    )


def test_bounded_training_advances_three_steps_exactly(tmp_path: Path) -> None:
    result = run_bounded_training(
        _config(tmp_path),
        bundle_store_path=tmp_path / "bundle",
        target_step=3,
        output_path=tmp_path / "result.json",
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    assert result.started_step == 0
    assert result.final_step == 3
    assert not result.resumed
    assert [item.committed_step for item in result.lineage] == [0, 1, 2, 3]
    assert [item.batch_cursor for item in result.steps] == [0, 1, 2]
    assert result.final_bounded_vs_resident_state.exact_bytes
    assert result.final_bundle_vs_restored_state.exact_bytes
    assert result.batch_cursor_derived_from_committed_step
    assert result.full_final_state_materialized_for_validation
    assert result.resident_reference_replayed_from_step_zero
    assert result.historical_bundles_retained
    assert all(item.resident_vs_candidate_state.exact_bytes for item in result.steps)
    assert all(item.candidate_vs_restored_state.exact_bytes for item in result.steps)
    assert all(item.full_candidate_state_materialized_for_validation for item in result.steps)
    assert all(item.resident_oracle_materialized_for_validation for item in result.steps)
    assert all(value == 3.0 for _, value in result.optimizer_step_values)
    assert StepBundleStore.open(tmp_path / "bundle").verify().committed_step == 3


def test_integrity_only_validation_skips_full_state_comparisons(
    tmp_path: Path,
) -> None:
    result = run_bounded_training(
        _config(tmp_path),
        bundle_store_path=tmp_path / "bundle",
        target_step=2,
        output_path=tmp_path / "result.json",
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
        validation_level="integrity_only",
    )

    assert result.validation_level == "integrity_only"
    assert result.final_step == 2
    assert result.final_bounded_vs_resident_state is None
    assert result.final_bundle_vs_restored_state is None
    assert not result.full_final_state_materialized_for_validation
    assert not result.resident_reference_replayed_from_step_zero
    assert result.validation_omitted_checks == (
        "final_bounded_vs_resident_state",
        "final_bundle_vs_restored_state",
        "resident_reference_replay_from_step_zero",
    )
    assert all(item.validation_level == "integrity_only" for item in result.steps)
    assert all(item.oracle_state_store_path is None for item in result.steps)
    assert all(item.resident_vs_candidate_state is None for item in result.steps)
    assert all(item.candidate_vs_restored_state is None for item in result.steps)
    assert not any(
        item.full_candidate_state_materialized_for_validation for item in result.steps
    )
    assert all(item.resident_oracle_materialized_for_validation for item in result.steps)
    assert all(value == 2.0 for _, value in result.optimizer_step_values)
    assert StepBundleStore.open(tmp_path / "bundle").verify().committed_step == 2


def test_validation_level_change_is_allowed_on_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle_path = tmp_path / "bundle"
    run_bounded_training(
        config,
        bundle_store_path=bundle_path,
        target_step=1,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    integrity_config = replace(
        config,
        training=replace(config.training, validation_level="integrity_only"),
    )

    result = run_bounded_training(
        integrity_config,
        bundle_store_path=bundle_path,
        target_step=2,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    assert result.resumed
    assert result.started_step == 1
    assert result.validation_level == "integrity_only"
    assert result.final_step == 2


def test_resumed_training_matches_uninterrupted_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_bounded_training(
        config,
        bundle_store_path=tmp_path / "resumed",
        target_step=2,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    resumed = run_bounded_training(
        config,
        bundle_store_path=tmp_path / "resumed",
        target_step=5,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    uninterrupted = run_bounded_training(
        config,
        bundle_store_path=tmp_path / "uninterrupted",
        target_step=5,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    assert first.final_step == 2
    assert first.final_bounded_vs_resident_state.exact_bytes
    assert resumed.resumed
    assert resumed.started_step == 2
    assert resumed.final_step == 5
    assert len(resumed.steps) == 3
    assert uninterrupted.final_step == 5
    comparison = compare_states(
        _current_state(tmp_path / "resumed"),
        _current_state(tmp_path / "uninterrupted"),
    )
    assert comparison.exact_bytes
    assert resumed.final_bounded_vs_resident_state.exact_bytes
    assert uninterrupted.final_bounded_vs_resident_state.exact_bytes
    assert all(value == 5.0 for _, value in resumed.optimizer_step_values)
    assert [item.committed_step for item in resumed.lineage] == list(range(6))


def test_resume_rejects_semantic_configuration_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_bounded_training(
        config,
        bundle_store_path=tmp_path / "bundle",
        target_step=1,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    changed = replace(
        config,
        training=replace(config.training, learning_rate=2e-3),
    )

    with pytest.raises(ResumeConfigurationError, match="config_digest"):
        run_bounded_training(
            changed,
            bundle_store_path=tmp_path / "bundle",
            target_step=2,
            device_override="cpu",
            optimizer_working_set_bytes=1024**2,
        )


def test_resume_detects_corrupt_current_child_store(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle_path = tmp_path / "bundle"
    run_bounded_training(
        config,
        bundle_store_path=bundle_path,
        target_step=1,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    bundle = StepBundleStore.open(bundle_path)
    current = bundle.current_manifest()
    parameter_store_path = bundle_path / current.parameter_store.path
    parameter_store = VersionedTensorStore.open(parameter_store_path)
    chunk = parameter_store.current_manifest().chunks[0]
    (parameter_store_path / chunk.storage_path).write_bytes(b"corrupt")

    with pytest.raises(IntegrityError):
        run_bounded_training(
            config,
            bundle_store_path=bundle_path,
            target_step=2,
            device_override="cpu",
            optimizer_working_set_bytes=1024**2,
        )


@pytest.mark.parametrize(
    "failure_point",
    [
        BundleFailurePoint.BEFORE_MANIFEST_RENAME,
        BundleFailurePoint.BEFORE_CURRENT_RENAME,
    ],
)
def test_later_bundle_failure_preserves_previous_step(
    tmp_path: Path,
    failure_point: BundleFailurePoint,
) -> None:
    config = _config(tmp_path)
    bundle_path = tmp_path / "bundle"
    run_bounded_training(
        config,
        bundle_store_path=bundle_path,
        target_step=2,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    def fail(point: BundleFailurePoint, context: dict[str, object]) -> None:
        del context
        if point is failure_point:
            raise BundleSimulatedCrash(f"stop at {point.value} during later step")

    with pytest.raises(BundleSimulatedCrash):
        run_bounded_training(
            config,
            bundle_store_path=bundle_path,
            target_step=3,
            device_override="cpu",
            optimizer_working_set_bytes=1024**2,
            bundle_failure_injector=fail,
        )

    bundle = StepBundleStore.open(bundle_path)
    assert bundle.current_manifest().committed_step == 2
    assert bundle.verify().committed_step == 2
    assert bundle.recover().current_bundle_id == bundle.current_manifest().bundle_id


def test_multi_step_optimizer_budget_rejection_preserves_step_zero(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    with pytest.raises(OptimizerWorkingSetExceededError):
        run_bounded_training(
            _config(tmp_path),
            bundle_store_path=bundle_path,
            target_step=1,
            device_override="cpu",
            optimizer_working_set_bytes=1,
        )

    bundle = StepBundleStore.open(bundle_path)
    assert bundle.current_manifest().committed_step == 0
    assert bundle.verify().committed_step == 0
