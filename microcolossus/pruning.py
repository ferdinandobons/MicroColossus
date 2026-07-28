"""Safe, explicit pruning of historical MicroColossus checkpoint state.

The data model and planning primitives live in :mod:`microcolossus.pruning_core`.
This public facade owns apply-time validation so benign read telemetry produced
after planning does not invalidate an otherwise unchanged checkpoint graph.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from . import pruning_core as _core
from .config import ExperimentConfig, RetentionConfig
from .pruning_core import (
    PRUNING_OPERATION_SCHEMA_VERSION,
    PRUNING_PLAN_SCHEMA_VERSION,
    PRUNING_REPORT_SCHEMA_VERSION,
    PruningFailureInjector,
    PruningFailurePoint,
    PruningInProgressError,
    PruningPathRecord,
    PruningPlan,
    PruningReport,
    PruningSimulatedCrash,
    assert_pruning_inactive,
    build_pruning_plan,
    load_pruning_plan,
    write_pruning_plan,
    write_pruning_report,
)
from .step_bundle import StepBundleStore
from .storage import IntegrityError

__all__ = [
    "PRUNING_OPERATION_SCHEMA_VERSION",
    "PRUNING_PLAN_SCHEMA_VERSION",
    "PRUNING_REPORT_SCHEMA_VERSION",
    "PruningFailureInjector",
    "PruningFailurePoint",
    "PruningInProgressError",
    "PruningPathRecord",
    "PruningPlan",
    "PruningReport",
    "PruningSimulatedCrash",
    "apply_pruning_plan",
    "assert_pruning_inactive",
    "build_pruning_plan",
    "load_pruning_plan",
    "write_pruning_plan",
    "write_pruning_report",
]


def _verify_fresh_plan(
    *,
    root: Path,
    plan: PruningPlan,
) -> tuple[StepBundleStore, tuple[str, ...]]:
    """Verify immutable authority and every planned deletion target.

    Read-only diagnostics can append tensor-store telemetry after a plan has
    been created. Telemetry growth does not change tensor manifests, root
    lineage, retention reachability, or deletion-target contents, so it must
    not invalidate the plan. The apply gate therefore compares semantic
    checkpoint topology and exact deletion inventories instead of rebuilding
    and byte-comparing the entire plan, whose managed-byte snapshot is
    intentionally time-dependent.
    """

    bundle_store = _core._verify_current(root, plan)
    manifests = _core._lineage_manifests(bundle_store)

    lineage_bundle_ids = tuple(item.bundle_id for item in manifests)
    lineage_steps = tuple(item.committed_step for item in manifests)
    if lineage_bundle_ids != plan.lineage_bundle_ids or lineage_steps != plan.lineage_steps:
        raise IntegrityError("root lineage changed after the pruning plan was created")

    policy = RetentionConfig(
        keep_previous=plan.keep_previous,
        milestone_interval=plan.milestone_interval,
    )
    retained = _core._retained_manifests(manifests, policy)
    retained_bundle_ids = tuple(item.bundle_id for item in retained)
    retained_steps = tuple(item.committed_step for item in retained)
    if retained_bundle_ids != plan.retained_bundle_ids or retained_steps != plan.retained_steps:
        raise IntegrityError("retained checkpoint set changed after planning")

    retained_store_paths = tuple(
        sorted(
            path
            for manifest in retained
            for path in _core._manifest_store_paths(manifest)
        )
    )
    if retained_store_paths != plan.retained_store_paths:
        raise IntegrityError("retained child-store reachability changed after planning")

    retained_ids = set(retained_bundle_ids)
    pruned = tuple(item for item in manifests if item.bundle_id not in retained_ids)
    pruned_bundle_ids = tuple(item.bundle_id for item in pruned)
    pruned_steps = tuple(item.committed_step for item in pruned)
    if (
        pruned_bundle_ids != plan.pruned_checkpoint_bundle_ids
        or pruned_steps != plan.pruned_checkpoint_steps
    ):
        raise IntegrityError("pruned checkpoint set changed after planning")

    verified = _core._verify_retained(bundle_store, plan)

    for record in plan.deletion_paths:
        target = _core._path_for(root, record.path)
        if not os.path.lexists(target):
            raise IntegrityError(
                f"pruning target disappeared before journal publication: {record.path}"
            )
        current_inventory = _core._path_inventory(
            root,
            record.path,
            record.category,
        )
        if current_inventory != record:
            raise IntegrityError(f"pruning target changed after planning: {record.path}")

    return bundle_store, verified


def apply_pruning_plan(
    config: ExperimentConfig,
    *,
    bundle_store_path: str | Path,
    plan: PruningPlan,
    failure_injector: PruningFailureInjector | None = None,
) -> PruningReport:
    """Apply one verified plan while keeping ``CURRENT`` byte-identical."""

    plan.validate()
    root = Path(bundle_store_path)
    _core._validate_config(root, config)
    operation_path = _core._operation_path(root, plan)

    with _core._PruningLock(root, plan.plan_checksum):
        if operation_path.exists():
            bundle_store = _core._verify_current(root, plan)
            verified = _core._verify_retained(bundle_store, plan)
            operation = _core._load_operation(operation_path)
            if operation.plan_checksum != plan.plan_checksum:
                raise IntegrityError("pruning journal references a different plan")
            if operation.current_bundle_id != plan.current_bundle_id:
                raise IntegrityError("pruning journal references a different CURRENT")
            if operation.state == "completed":
                recovery = bundle_store.recover()
                return _core._report(
                    plan,
                    operation,
                    root=root,
                    verified=verified,
                    recovery=recovery,
                    newly_reclaimed_bytes=0,
                    idempotent=True,
                )
        else:
            bundle_store, verified = _verify_fresh_plan(
                root=root,
                plan=plan,
            )
            operation = _core._PruningOperation(
                schema_version=PRUNING_OPERATION_SCHEMA_VERSION,
                plan_checksum=plan.plan_checksum,
                current_bundle_id=plan.current_bundle_id,
                state="prepared",
                completed_paths=(),
                already_missing_paths=(),
                cumulative_reclaimed_bytes=0,
                managed_bytes_after=None,
            )

            def before_journal_rename() -> None:
                if failure_injector is not None:
                    failure_injector(
                        PruningFailurePoint.BEFORE_JOURNAL_RENAME,
                        {"plan_checksum": plan.plan_checksum},
                    )

            _core._write_operation(
                operation_path,
                operation,
                before_rename=before_journal_rename,
            )
            operation = _core._load_operation(operation_path)

        operation = replace(operation, state="deleting")
        _core._write_operation(operation_path, operation)
        completed = list(operation.completed_paths)
        missing = list(operation.already_missing_paths)
        reclaimed = operation.cumulative_reclaimed_bytes
        newly_reclaimed = 0
        remaining = [item for item in plan.deletion_paths if item.path not in completed]
        if remaining and failure_injector is not None:
            failure_injector(
                PruningFailurePoint.BEFORE_FIRST_DELETE,
                {"path": remaining[0].path},
            )

        for record in remaining:
            _core._verify_current(root, plan)
            target = _core._path_for(root, record.path)
            if not os.path.lexists(target):
                completed.append(record.path)
                missing.append(record.path)
                reclaimed += record.byte_count
                newly_reclaimed += record.byte_count
            else:
                current_inventory = _core._path_inventory(
                    root,
                    record.path,
                    record.category,
                )
                if current_inventory != record:
                    raise IntegrityError(
                        f"pruning target changed after planning: {record.path}"
                    )
                _core._delete_path(target)
                if failure_injector is not None:
                    failure_injector(
                        PruningFailurePoint.AFTER_PATH_DELETE,
                        {"path": record.path},
                    )
                completed.append(record.path)
                reclaimed += record.byte_count
                newly_reclaimed += record.byte_count
            operation = replace(
                operation,
                state="deleting",
                completed_paths=tuple(completed),
                already_missing_paths=tuple(missing),
                cumulative_reclaimed_bytes=reclaimed,
            )
            _core._write_operation(operation_path, operation)

        bundle_store = _core._verify_current(root, plan)
        verified = _core._verify_retained(bundle_store, plan)
        recovery = bundle_store.recover()
        managed_after = _core._managed_size(root)
        operation = replace(
            operation,
            state="completed",
            managed_bytes_after=managed_after,
        )
        _core._write_operation(operation_path, operation)
        completed_operation = _core._load_operation(operation_path)
        report = _core._report(
            plan,
            completed_operation,
            root=root,
            verified=verified,
            recovery=recovery,
            newly_reclaimed_bytes=newly_reclaimed,
            idempotent=False,
        )
        if not report.current_pointer_unchanged:
            raise IntegrityError("CURRENT changed during checkpoint pruning")
        return report
