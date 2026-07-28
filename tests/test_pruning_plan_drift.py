from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.bounded_optimizer import _store_payloads
from microcolossus.bounded_training import run_bounded_training
from microcolossus.config import load_experiment_config
from microcolossus.pruning import (
    PruningFailurePoint,
    PruningSimulatedCrash,
    apply_pruning_plan,
    build_pruning_plan,
)
from microcolossus.step_bundle import StepBundleStore
from microcolossus.storage import IntegrityError
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


def _managed_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != "pruning"
    )


def _read_current_state(root: Path) -> None:
    bundle_store = StepBundleStore.open(root)
    current = bundle_store.current_manifest()
    parameter_store = _open_referenced_store(
        bundle_store,
        current,
        kind="parameter",
    )
    optimizer_store = _open_referenced_store(
        bundle_store,
        current,
        kind="optimizer",
    )
    _store_payloads(parameter_store)
    _store_payloads(optimizer_store)


def test_apply_allows_retained_read_telemetry_after_planning(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 5)
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=1,
        milestone_interval=0,
    )
    current_before = (root / "CURRENT").read_bytes()
    managed_before_read = _managed_bytes(root)

    _read_current_state(root)

    assert _managed_bytes(root) > managed_before_read
    report = apply_pruning_plan(
        _config(),
        bundle_store_path=root,
        plan=plan,
    )
    assert report.current_pointer_unchanged
    assert report.current_step == 5
    assert report.retained_steps == (4, 5)
    assert report.cumulative_reclaimed_bytes == plan.selected_bytes
    assert (root / "CURRENT").read_bytes() == current_before


def test_pre_journal_retry_allows_retained_read_telemetry(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 3)
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=0,
        milestone_interval=0,
    )
    current_before = (root / "CURRENT").read_bytes()

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

    _read_current_state(root)
    completed = apply_pruning_plan(
        _config(),
        bundle_store_path=root,
        plan=plan,
    )
    assert completed.current_pointer_unchanged
    assert completed.cumulative_reclaimed_bytes == plan.selected_bytes
    assert (root / "CURRENT").read_bytes() == current_before


def test_fresh_apply_rejects_changed_deletion_target(tmp_path: Path) -> None:
    root = tmp_path / "training"
    _train(root, 3)
    plan = build_pruning_plan(
        _config(),
        bundle_store_path=root,
        keep_previous=0,
        milestone_interval=0,
    )
    target = root / plan.deletion_paths[0].path
    assert target.is_dir()
    (target / "unexpected-after-plan.bin").write_bytes(b"changed")

    with pytest.raises(IntegrityError, match="pruning target changed after planning"):
        apply_pruning_plan(
            _config(),
            bundle_store_path=root,
            plan=plan,
        )

    assert target.exists()
    StepBundleStore.open(root).verify()
