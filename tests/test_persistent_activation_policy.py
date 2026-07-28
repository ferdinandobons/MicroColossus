from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from microcolossus.activation_recompute import (
    ActivationWorkingSetExceededError,
    WorkspaceWorkingSetExceededError,
)
from microcolossus.bounded_training import ResumeConfigurationError, run_bounded_training
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.pruning import apply_pruning_plan, build_pruning_plan
from microcolossus.step_bundle import StepBundleStore
from microcolossus.storage import VersionedTensorStore
from microcolossus.storage_training import compare_states


def _config(tmp_path: Path, *, activation_policy: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"persistent-activation-{activation_policy}",
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
            steps=4,
            micro_batch_size=1,
            sequence_length=4,
            learning_rate=1e-3,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
            seed=29,
            device="cpu",
            activation_policy=activation_policy,
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
    stores = (
        VersionedTensorStore.open(path / current.parameter_store.path),
        VersionedTensorStore.open(path / current.optimizer_store.path),
    )
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


def _run(
    config: ExperimentConfig,
    path: Path,
    target_step: int,
    *,
    activation_bytes: int = 1024**2,
    workspace_bytes: int = 4 * 1024**2,
):
    return run_bounded_training(
        config,
        bundle_store_path=path,
        target_step=target_step,
        device_override="cpu",
        parameter_working_set_bytes=1024**2,
        gradient_working_set_bytes=1024**2,
        optimizer_working_set_bytes=1024**2,
        activation_working_set_bytes=activation_bytes,
        workspace_working_set_bytes=workspace_bytes,
    )


def test_recompute_multi_step_matches_retain_all_exactly(tmp_path: Path) -> None:
    retained = _run(
        _config(tmp_path, activation_policy="retain_all"),
        tmp_path / "retained",
        3,
    )
    recomputed = _run(
        _config(tmp_path, activation_policy="recompute"),
        tmp_path / "recomputed",
        3,
    )

    comparison = compare_states(
        _current_state(tmp_path / "retained"),
        _current_state(tmp_path / "recomputed"),
    )
    assert comparison.exact_bytes
    assert recomputed.activation_policy == "recompute"
    assert recomputed.maximum_retained_forward_boundary_bytes == 0
    assert recomputed.maximum_retained_activation_bytes > 0
    assert recomputed.maximum_workspace_bytes > 0
    assert recomputed.total_prefix_replayed_groups == 9
    assert recomputed.total_prefix_recomputation_seconds > 0.0
    assert [item.batch_checksum for item in retained.steps] == [
        item.batch_checksum for item in recomputed.steps
    ]
    assert all(item.activation_policy == "recompute" for item in recomputed.steps)
    assert all(item.retained_forward_boundary_count == 0 for item in recomputed.steps)
    assert all(item.retained_forward_boundary_bytes == 0 for item in recomputed.steps)
    assert all(item.total_prefix_replayed_groups == 3 for item in recomputed.steps)
    assert all(item.activation_budget_respected for item in recomputed.steps)
    assert all(item.workspace_budget_respected for item in recomputed.steps)
    assert all(item.resident_vs_candidate_state.exact_bytes for item in recomputed.steps)
    assert all(item.candidate_vs_restored_state.exact_bytes for item in recomputed.steps)
    assert recomputed.final_bounded_vs_resident_state.exact_bytes
    assert recomputed.final_bundle_vs_restored_state.exact_bytes


def test_recompute_process_resume_matches_uninterrupted(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="recompute")
    first = _run(config, tmp_path / "resumed", 1)
    resumed = _run(config, tmp_path / "resumed", 3)
    uninterrupted = _run(config, tmp_path / "uninterrupted", 3)

    assert first.final_step == 1
    assert resumed.resumed
    assert resumed.started_step == 1
    assert resumed.final_step == 3
    assert compare_states(
        _current_state(tmp_path / "resumed"),
        _current_state(tmp_path / "uninterrupted"),
    ).exact_bytes
    assert resumed.final_bounded_vs_resident_state.exact_bytes
    assert uninterrupted.final_bounded_vs_resident_state.exact_bytes
    assert [item.committed_step for item in resumed.lineage] == [0, 1, 2, 3]


def test_resume_rejects_activation_policy_change(tmp_path: Path) -> None:
    retained = _config(tmp_path, activation_policy="retain_all")
    _run(retained, tmp_path / "bundle", 1)
    recompute = replace(
        retained,
        training=replace(retained.training, activation_policy="recompute"),
    )

    with pytest.raises(ResumeConfigurationError, match="config_digest"):
        _run(recompute, tmp_path / "bundle", 2)
    assert StepBundleStore.open(tmp_path / "bundle").current_manifest().committed_step == 1


@pytest.mark.parametrize(
    ("activation_bytes", "workspace_bytes", "expected"),
    [
        (1, 4 * 1024**2, ActivationWorkingSetExceededError),
        (1024**2, 1, WorkspaceWorkingSetExceededError),
    ],
)
def test_recompute_budget_rejection_preserves_step_zero(
    tmp_path: Path,
    activation_bytes: int,
    workspace_bytes: int,
    expected: type[RuntimeError],
) -> None:
    bundle_path = tmp_path / f"bundle-{expected.__name__}"
    with pytest.raises(expected):
        _run(
            _config(tmp_path, activation_policy="recompute"),
            bundle_path,
            1,
            activation_bytes=activation_bytes,
            workspace_bytes=workspace_bytes,
        )

    bundle = StepBundleStore.open(bundle_path)
    assert bundle.current_manifest().committed_step == 0
    assert bundle.verify().committed_step == 0


def test_recompute_resume_after_pruning_matches_unpruned(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="recompute")
    pruned_path = tmp_path / "pruned"
    _run(config, pruned_path, 3)
    plan = build_pruning_plan(
        config,
        bundle_store_path=pruned_path,
        keep_previous=0,
        milestone_interval=0,
    )
    report = apply_pruning_plan(
        config,
        bundle_store_path=pruned_path,
        plan=plan,
    )
    assert report.retained_steps == (3,)
    assert report.current_pointer_unchanged

    resumed = _run(config, pruned_path, 4)
    uninterrupted = _run(config, tmp_path / "unpruned", 4)
    assert resumed.resumed
    assert resumed.started_step == 3
    assert compare_states(
        _current_state(pruned_path),
        _current_state(tmp_path / "unpruned"),
    ).exact_bytes
    assert resumed.final_bundle_vs_restored_state.exact_bytes
    assert uninterrupted.final_bundle_vs_restored_state.exact_bytes
