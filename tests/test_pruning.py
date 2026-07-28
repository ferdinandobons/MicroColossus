from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from microcolossus.bounded_optimizer import _store_payloads
from microcolossus.bounded_training import run_bounded_training
from microcolossus.config import RetentionConfig, load_experiment_config
from microcolossus.pruning import (
    PRUNING_PLAN_SCHEMA_VERSION,
    PruningFailurePoint,
    PruningInProgressError,
    PruningSimulatedCrash,
    apply_pruning_plan,
    build_pruning_plan,
)
from microcolossus.step_bundle import StepBundleStore
from microcolossus.storage import IntegrityError, VersionedTensorStore
from microcolossus.storage_training import compare_states
from microcolossus.training_checkpoint import _open_referenced_store


def _config():
    return load_experiment_config(Path("examples/micro-storage.yaml"))


def _train(root: Path, step: int) -> None:
    run_bounded_training(
        _config(),
        bundle_store_path=root,
        target_step=step,
        device_override="cpu",
    )


def _current_state(root: Path):
    store = StepBundleStore.open(root)
    current = store.current_manifest()
    parameter_store = _open_referenced_store(store, current, kind="parameter")
    optimizer_store = _open_referenced_store(store, current, kind="optimizer")
    return tuple(
        sorted(
            _store_payloads(parameter_store) + _store_payloads(optimizer_store),
            key=lambda item: item.logical_name,
        )
    )


def test_retention_configuration_validation() -> None:
    assert RetentionConfig().keep_previous == 2
    assert RetentionConfig(milestone_interval=5).milestone_interval == 5
    with pytest.raises(ValueError, match="keep_previous"):
        RetentionConfig(keep_previous=-1)
    with pytest.raises(ValueError, match="milestone_interval"):
        RetentionConfig(milestone_interval=-1)


def test_pruning_plan_is_deterministic_and_non_mutating(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 5)
    current_before = (root / "CURRENT").read_bytes()
    size_before = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    first = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=1,
        milestone_interval=3,
    )
    second = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=1,
        milestone_interval=3,
    )

    assert first.schema_version == PRUNING_PLAN_SCHEMA_VERSION
    assert first.to_dict() == second.to_dict()
    assert first.retained_steps == (0, 3, 4, 5)
    assert first.pruned_checkpoint_steps == (1, 2)
    assert first.selected_bytes > 0
    assert first.deletion_paths
    assert (root / "CURRENT").read_bytes() == current_before
    size_after = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert size_after == size_before


def test_apply_pruning_preserves_current_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 5)
    current_before = (root / "CURRENT").read_bytes()
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=1,
        milestone_interval=0,
    )

    report = apply_pruning_plan(
        _config(),
        bundle_store_path=root,
        plan=plan,
    )

    assert report.current_pointer_unchanged
    assert report.current_step == 5
    assert report.retained_steps == (4, 5)
    assert report.cumulative_reclaimed_bytes == plan.selected_bytes
    assert report.newly_reclaimed_bytes == plan.selected_bytes
    assert report.managed_bytes_after < report.managed_bytes_before
    assert not report.idempotent
    assert (root / "CURRENT").read_bytes() == current_before
    store = StepBundleStore.open(root)
    for bundle_id in plan.retained_bundle_ids:
        store.verify(bundle_id)

    repeated = apply_pruning_plan(
        _config(),
        bundle_store_path=root,
        plan=plan,
    )
    assert repeated.idempotent
    assert repeated.newly_reclaimed_bytes == 0
    assert repeated.cumulative_reclaimed_bytes == report.cumulative_reclaimed_bytes
    assert (root / "CURRENT").read_bytes() == current_before


def test_resume_after_pruning_matches_unpruned_training(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    pruned = tmp_path / "pruned"
    _train(reference, 4)
    _train(pruned, 4)

    plan = build_pruning_plan(
        _config(),
        bundle_store_path=pruned,
        keep_previous=0,
        milestone_interval=0,
    )
    apply_pruning_plan(_config(), bundle_store_path=pruned, plan=plan)

    _train(reference, 5)
    _train(pruned, 5)
    comparison = compare_states(_current_state(reference), _current_state(pruned))
    assert comparison.exact_bytes
    assert comparison.maximum_absolute_difference == 0.0


def test_interrupted_pruning_resumes_without_harming_retained_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "training"
    _train(root, 4)
    current_before = (root / "CURRENT").read_bytes()
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=0,
        milestone_interval=0,
    )
    triggered = False

    def injector(point: PruningFailurePoint, context: dict[str, object]) -> None:
        nonlocal triggered
        if point is PruningFailurePoint.AFTER_PATH_DELETE and not triggered:
            triggered = True
            raise PruningSimulatedCrash(str(context["path"]))

    with pytest.raises(PruningSimulatedCrash):
        apply_pruning_plan(
            _config(),
            bundle_store_path=root,
            plan=plan,
            failure_injector=injector,
        )

    assert (root / "CURRENT").read_bytes() == current_before
    StepBundleStore.open(root).verify()
    completed = apply_pruning_plan(_config(), bundle_store_path=root, plan=plan)
    assert completed.current_pointer_unchanged
    assert completed.already_missing_paths
    assert completed.cumulative_reclaimed_bytes == plan.selected_bytes
    StepBundleStore.open(root).verify()


def test_failure_before_journal_publication_does_not_delete_state(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 3)
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=0,
        milestone_interval=0,
    )
    first_target = root / plan.deletion_paths[0].path

    def injector(point: PruningFailurePoint, context: dict[str, object]) -> None:
        del context
        if point is PruningFailurePoint.BEFORE_JOURNAL_RENAME:
            raise PruningSimulatedCrash(point.value)

    with pytest.raises(PruningSimulatedCrash):
        apply_pruning_plan(
            _config(),
            bundle_store_path=root,
            plan=plan,
            failure_injector=injector,
        )
    assert first_target.exists()
    StepBundleStore.open(root).verify()


def test_corrupt_retained_child_blocks_pruning_before_deletion(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 3)
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=0,
        milestone_interval=0,
    )
    first_target = root / plan.deletion_paths[0].path
    store = StepBundleStore.open(root)
    current = store.current_manifest()
    parameter_store = VersionedTensorStore.open(root / current.parameter_store.path)
    chunk = parameter_store.current_manifest().chunks[0]
    chunk_path = parameter_store.root / chunk.storage_path
    chunk_path.write_bytes(b"corrupt")

    with pytest.raises(IntegrityError):
        apply_pruning_plan(_config(), bundle_store_path=root, plan=plan)
    assert first_target.exists()


def test_unreferenced_corrupt_orphan_is_safely_removed(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 2)
    orphan = root / "candidates" / "orphan-corrupt"
    orphan.mkdir()
    (orphan / "invalid.bin").write_bytes(b"not a tensor store")

    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=0,
        milestone_interval=0,
    )
    assert any(item.path == "candidates/orphan-corrupt" for item in plan.deletion_paths)
    apply_pruning_plan(_config(), bundle_store_path=root, plan=plan)
    assert not orphan.exists()
    StepBundleStore.open(root).verify()


def test_training_rejects_active_pruning_lock(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 1)
    lock = root / "pruning" / "LOCK"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps({"pid": os.getpid(), "plan_checksum": "test"}),
        encoding="utf-8",
    )

    with pytest.raises(PruningInProgressError):
        run_bounded_training(
            _config(),
            bundle_store_path=root,
            target_step=2,
            device_override="cpu",
        )


def test_retention_policy_is_not_part_of_training_semantics(tmp_path: Path) -> None:
    root = tmp_path / "training"
    config = _config()
    _train(root, 1)
    changed = replace(
        config,
        retention=RetentionConfig(keep_previous=0, milestone_interval=7),
    )
    result = run_bounded_training(
        changed,
        bundle_store_path=root,
        target_step=2,
        device_override="cpu",
    )
    assert result.resumed
    assert result.final_step == 2
