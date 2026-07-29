from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from microcolossus.activation_recompute import (
    ActivationWorkingSetExceededError,
    WorkspaceWorkingSetExceededError,
)
from microcolossus.activation_planner import ActivationPlanIntegrityError
from microcolossus.bounded_training import ResumeConfigurationError, run_bounded_training
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
    load_experiment_config,
)
from microcolossus.pruning import apply_pruning_plan, build_pruning_plan
from microcolossus.step_bundle import (
    BundleFailurePoint,
    BundleSimulatedCrash,
    StepBundleStore,
)
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

    backward = json.loads(
        Path(recomputed.steps[0].bounded_backward_result_path).read_text(encoding="utf-8")
    )
    assert backward["tied_gradient_accumulation_count"] == 2
    assert backward["tied_gradient_version"] == 1
    assert backward["backward_group_order"] == ["final-head", "block-0", "embedding"]
    assert backward["backward_groups"][0]["prefix_replay"]["replayed_group_names"] == [
        "embedding",
        "block-0",
    ]
    assert backward["backward_groups"][1]["prefix_replay"]["replayed_group_names"] == [
        "embedding"
    ]
    assert backward["backward_groups"][2]["prefix_replay"] is None


def test_real_text_micro_recompute_matches_retain_all(tmp_path: Path) -> None:
    retained_config = load_experiment_config("examples/real-text-micro.yaml")
    recompute_config = replace(
        retained_config,
        name="real-text-micro-recompute-test",
        training=replace(retained_config.training, activation_policy="recompute"),
    )
    retained = _run(retained_config, tmp_path / "real-retained", 2)
    recomputed = _run(recompute_config, tmp_path / "real-recomputed", 2)

    assert compare_states(
        _current_state(tmp_path / "real-retained"),
        _current_state(tmp_path / "real-recomputed"),
    ).exact_bytes
    assert [item.batch_checksum for item in retained.steps] == [
        item.batch_checksum for item in recomputed.steps
    ]
    assert [item.batch_cursor for item in retained.steps] == [
        item.batch_cursor for item in recomputed.steps
    ]
    assert recomputed.maximum_retained_forward_boundary_bytes == 0
    assert recomputed.total_prefix_replayed_groups == 6
    assert recomputed.final_bundle_vs_restored_state.exact_bytes


def test_hybrid_multi_step_is_intermediate_and_matches_extremes(
    tmp_path: Path,
) -> None:
    retained = _run(
        _config(tmp_path, activation_policy="retain_all"),
        tmp_path / "hybrid-retained",
        3,
    )
    recomputed = _run(
        _config(tmp_path, activation_policy="recompute"),
        tmp_path / "hybrid-recomputed",
        3,
    )
    hybrid = _run(
        _config(tmp_path, activation_policy="hybrid"),
        tmp_path / "hybrid",
        3,
        activation_bytes=512,
    )

    assert hybrid.activation_policy == "hybrid"
    assert all(item.activation_policy == "hybrid" for item in hybrid.steps)
    assert compare_states(
        _current_state(tmp_path / "hybrid-retained"),
        _current_state(tmp_path / "hybrid"),
    ).exact_bytes
    assert compare_states(
        _current_state(tmp_path / "hybrid-recomputed"),
        _current_state(tmp_path / "hybrid"),
    ).exact_bytes
    assert 0 < hybrid.maximum_retained_forward_boundary_bytes < (
        retained.maximum_retained_forward_boundary_bytes
    )
    assert 0 < hybrid.total_prefix_replayed_groups < (
        recomputed.total_prefix_replayed_groups
    )
    assert hybrid.total_prefix_parameter_logical_bytes_read < (
        recomputed.total_prefix_parameter_logical_bytes_read
    )
    assert hybrid.final_bounded_vs_resident_state.exact_bytes
    assert hybrid.final_bundle_vs_restored_state.exact_bytes
    assert all(item.resident_vs_candidate_state.exact_bytes for item in hybrid.steps)
    assert all(item.candidate_vs_restored_state.exact_bytes for item in hybrid.steps)

    metadata = hybrid.training_metadata
    assert metadata.activation_plan_checksum is not None
    assert metadata.activation_profile_checksum is not None
    assert metadata.activation_budget_bytes == 512
    assert metadata.workspace_budget_bytes == 4 * 1024**2
    assert metadata.activation_anchor_group_names

    backward = json.loads(
        Path(hybrid.steps[0].bounded_backward_result_path).read_text(encoding="utf-8")
    )
    assert backward["activation_policy"] == "hybrid"
    assert backward["selected_anchor_group_names"] == list(
        metadata.activation_anchor_group_names
    )
    assert backward["tied_gradient_accumulation_count"] == 2
    assert backward["backward_group_order"] == ["final-head", "block-0", "embedding"]
    assert backward["backward_groups"][0]["prefix_replay"]["replayed_group_names"] == []
    assert backward["backward_groups"][1]["prefix_replay"]["replayed_group_names"] == [
        "embedding"
    ]
    assert backward["backward_groups"][2]["prefix_replay"] is None


def test_hybrid_resume_rejects_changed_budget_or_plan(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="hybrid")
    _run(config, tmp_path / "hybrid-plan-root", 1, activation_bytes=512)

    with pytest.raises(ResumeConfigurationError, match="activation_plan_checksum"):
        _run(config, tmp_path / "hybrid-plan-root", 2, activation_bytes=768)
    assert (
        StepBundleStore.open(tmp_path / "hybrid-plan-root")
        .current_manifest()
        .committed_step
        == 1
    )


def test_hybrid_resume_rejects_tampered_plan_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="hybrid")
    bundle_path = tmp_path / "hybrid-tampered-plan"
    _run(config, bundle_path, 1, activation_bytes=512)
    plan_path = bundle_path / "HYBRID_ACTIVATION_PLAN.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["total_replayed_groups"] += 1
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResumeConfigurationError, match="hybrid activation"):
        _run(config, bundle_path, 2, activation_bytes=512)
    assert StepBundleStore.open(bundle_path).current_manifest().committed_step == 1


def test_hybrid_infeasible_plan_creates_no_root(tmp_path: Path) -> None:
    destination = tmp_path / "hybrid-infeasible"

    with pytest.raises(ActivationPlanIntegrityError):
        _run(
            _config(tmp_path, activation_policy="hybrid"),
            destination,
            1,
            activation_bytes=1,
        )

    assert not destination.exists()


def test_hybrid_process_resume_matches_uninterrupted(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="hybrid")
    first = _run(config, tmp_path / "hybrid-resumed", 1, activation_bytes=512)
    resumed = _run(config, tmp_path / "hybrid-resumed", 3, activation_bytes=512)
    uninterrupted = _run(
        config,
        tmp_path / "hybrid-uninterrupted",
        3,
        activation_bytes=512,
    )

    assert first.final_step == 1
    assert resumed.resumed
    assert resumed.started_step == 1
    assert resumed.final_step == 3
    assert compare_states(
        _current_state(tmp_path / "hybrid-resumed"),
        _current_state(tmp_path / "hybrid-uninterrupted"),
    ).exact_bytes
    assert resumed.final_bounded_vs_resident_state.exact_bytes
    assert uninterrupted.final_bounded_vs_resident_state.exact_bytes
    assert [item.committed_step for item in resumed.lineage] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "failure_point",
    [
        BundleFailurePoint.BEFORE_MANIFEST_RENAME,
        BundleFailurePoint.BEFORE_CURRENT_RENAME,
    ],
)
def test_hybrid_later_failure_preserves_previous_bundle(
    tmp_path: Path,
    failure_point: BundleFailurePoint,
) -> None:
    config = _config(tmp_path, activation_policy="hybrid")
    bundle_path = tmp_path / f"hybrid-failure-{failure_point.value}"
    _run(config, bundle_path, 1, activation_bytes=512)

    def fail(point: BundleFailurePoint, context: dict[str, object]) -> None:
        del context
        if point is failure_point:
            raise BundleSimulatedCrash("stop hybrid publication")

    with pytest.raises(BundleSimulatedCrash):
        run_bounded_training(
            config,
            bundle_store_path=bundle_path,
            target_step=2,
            device_override="cpu",
            parameter_working_set_bytes=1024**2,
            gradient_working_set_bytes=1024**2,
            optimizer_working_set_bytes=1024**2,
            activation_working_set_bytes=512,
            workspace_working_set_bytes=4 * 1024**2,
            bundle_failure_injector=fail,
        )

    bundle = StepBundleStore.open(bundle_path)
    assert bundle.current_manifest().committed_step == 1
    assert bundle.verify().committed_step == 1
    assert bundle.recover().current_bundle_id == bundle.current_manifest().bundle_id


def test_hybrid_resume_after_pruning_matches_unpruned(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="hybrid")
    pruned_path = tmp_path / "hybrid-pruned"
    _run(config, pruned_path, 3, activation_bytes=512)
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

    resumed = _run(config, pruned_path, 4, activation_bytes=512)
    uninterrupted = _run(
        config,
        tmp_path / "hybrid-unpruned",
        4,
        activation_bytes=512,
    )
    assert resumed.resumed
    assert resumed.started_step == 3
    assert compare_states(
        _current_state(pruned_path),
        _current_state(tmp_path / "hybrid-unpruned"),
    ).exact_bytes
    assert resumed.final_bundle_vs_restored_state.exact_bytes
    assert uninterrupted.final_bundle_vs_restored_state.exact_bytes


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


def test_recompute_later_failure_preserves_previous_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path, activation_policy="recompute")
    bundle_path = tmp_path / "failure-bundle"
    _run(config, bundle_path, 1)

    def fail(point: BundleFailurePoint, context: dict[str, object]) -> None:
        del context
        if point is BundleFailurePoint.BEFORE_CURRENT_RENAME:
            raise BundleSimulatedCrash("stop recompute publication before CURRENT")

    with pytest.raises(BundleSimulatedCrash):
        run_bounded_training(
            config,
            bundle_store_path=bundle_path,
            target_step=2,
            device_override="cpu",
            parameter_working_set_bytes=1024**2,
            gradient_working_set_bytes=1024**2,
            optimizer_working_set_bytes=1024**2,
            activation_working_set_bytes=1024**2,
            workspace_working_set_bytes=4 * 1024**2,
            bundle_failure_injector=fail,
        )

    bundle = StepBundleStore.open(bundle_path)
    assert bundle.current_manifest().committed_step == 1
    assert bundle.verify().committed_step == 1
    assert bundle.recover().current_bundle_id == bundle.current_manifest().bundle_id


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
