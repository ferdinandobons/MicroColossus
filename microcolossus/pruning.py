"""Safe, explicit pruning of historical MicroColossus checkpoint state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ExperimentConfig, RetentionConfig
from .step_bundle import BundleRecoveryReport, StepBundleManifest, StepBundleStore
from .storage import IntegrityError
from .storage.schema import canonical_json_bytes, sha256_hex
from .training_checkpoint import (
    ResumeConfigurationError,
    _lineage,
    _load_training_metadata,
    config_digest,
)

PRUNING_PLAN_SCHEMA_VERSION = "microcolossus.pruning-plan.v1"
PRUNING_OPERATION_SCHEMA_VERSION = "microcolossus.pruning-operation.v1"
PRUNING_REPORT_SCHEMA_VERSION = "microcolossus.pruning-report.v1"


class PruningFailurePoint(StrEnum):
    BEFORE_JOURNAL_RENAME = "before_journal_rename"
    BEFORE_FIRST_DELETE = "before_first_delete"
    AFTER_PATH_DELETE = "after_path_delete"


class PruningSimulatedCrash(RuntimeError):
    """Raised by failure-injection tests during pruning."""


class PruningInProgressError(RuntimeError):
    """Raised when training or another pruning process owns the root."""


PruningFailureInjector = Callable[[PruningFailurePoint, dict[str, Any]], None]


@dataclass(frozen=True)
class PruningPathRecord:
    path: str
    category: str
    byte_count: int
    file_count: int
    content_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PruningPathRecord:
        return cls(
            path=str(value["path"]),
            category=str(value["category"]),
            byte_count=int(value["byte_count"]),
            file_count=int(value["file_count"]),
            content_checksum=str(value["content_checksum"]),
        )


@dataclass(frozen=True)
class PruningPlan:
    schema_version: str
    current_bundle_id: str
    current_bundle_checksum: str
    current_pointer_checksum: str
    current_step: int
    keep_previous: int
    milestone_interval: int
    lineage_bundle_ids: tuple[str, ...]
    lineage_steps: tuple[int, ...]
    retained_bundle_ids: tuple[str, ...]
    retained_steps: tuple[int, ...]
    retained_store_paths: tuple[str, ...]
    pruned_checkpoint_bundle_ids: tuple[str, ...]
    pruned_checkpoint_steps: tuple[int, ...]
    deletion_paths: tuple[PruningPathRecord, ...]
    managed_bytes_before: int
    selected_bytes: int
    plan_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "current_bundle_id": self.current_bundle_id,
            "current_bundle_checksum": self.current_bundle_checksum,
            "current_pointer_checksum": self.current_pointer_checksum,
            "current_step": self.current_step,
            "keep_previous": self.keep_previous,
            "milestone_interval": self.milestone_interval,
            "lineage_bundle_ids": list(self.lineage_bundle_ids),
            "lineage_steps": list(self.lineage_steps),
            "retained_bundle_ids": list(self.retained_bundle_ids),
            "retained_steps": list(self.retained_steps),
            "retained_store_paths": list(self.retained_store_paths),
            "pruned_checkpoint_bundle_ids": list(self.pruned_checkpoint_bundle_ids),
            "pruned_checkpoint_steps": list(self.pruned_checkpoint_steps),
            "deletion_paths": [item.to_dict() for item in self.deletion_paths],
            "managed_bytes_before": self.managed_bytes_before,
            "selected_bytes": self.selected_bytes,
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_checksum(self) -> PruningPlan:
        return replace(self, plan_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.schema_version != PRUNING_PLAN_SCHEMA_VERSION:
            raise IntegrityError(f"unsupported pruning plan schema: {self.schema_version}")
        RetentionConfig(
            keep_previous=self.keep_previous,
            milestone_interval=self.milestone_interval,
        )
        if self.plan_checksum != self.compute_checksum():
            raise IntegrityError("pruning plan checksum mismatch")
        if not self.lineage_steps or self.lineage_steps[-1] != self.current_step:
            raise IntegrityError("pruning plan lineage does not end at CURRENT")
        if len(self.lineage_bundle_ids) != len(self.lineage_steps):
            raise IntegrityError("pruning plan lineage fields have different lengths")
        if len(self.retained_bundle_ids) != len(self.retained_steps):
            raise IntegrityError("pruning plan retained fields have different lengths")
        if self.current_bundle_id not in self.retained_bundle_ids:
            raise IntegrityError("pruning plan does not retain CURRENT")
        if self.current_step not in self.retained_steps:
            raise IntegrityError("pruning plan does not retain the current step")
        if self.selected_bytes != sum(item.byte_count for item in self.deletion_paths):
            raise IntegrityError("pruning plan selected byte count is inconsistent")
        paths = [item.path for item in self.deletion_paths]
        if len(paths) != len(set(paths)):
            raise IntegrityError("pruning plan contains duplicate deletion paths")
        for index, path in enumerate(paths):
            _validate_deletion_path(path)
            for other in paths[index + 1 :]:
                if _is_ancestor(path, other) or _is_ancestor(other, path):
                    raise IntegrityError("pruning plan contains overlapping deletion paths")

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["plan_checksum"] = self.plan_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PruningPlan:
        result = cls(
            schema_version=str(value["schema_version"]),
            current_bundle_id=str(value["current_bundle_id"]),
            current_bundle_checksum=str(value["current_bundle_checksum"]),
            current_pointer_checksum=str(value["current_pointer_checksum"]),
            current_step=int(value["current_step"]),
            keep_previous=int(value["keep_previous"]),
            milestone_interval=int(value["milestone_interval"]),
            lineage_bundle_ids=tuple(str(item) for item in value["lineage_bundle_ids"]),
            lineage_steps=tuple(int(item) for item in value["lineage_steps"]),
            retained_bundle_ids=tuple(
                str(item) for item in value["retained_bundle_ids"]
            ),
            retained_steps=tuple(int(item) for item in value["retained_steps"]),
            retained_store_paths=tuple(
                str(item) for item in value["retained_store_paths"]
            ),
            pruned_checkpoint_bundle_ids=tuple(
                str(item) for item in value["pruned_checkpoint_bundle_ids"]
            ),
            pruned_checkpoint_steps=tuple(
                int(item) for item in value["pruned_checkpoint_steps"]
            ),
            deletion_paths=tuple(
                PruningPathRecord.from_dict(dict(item))
                for item in value["deletion_paths"]
            ),
            managed_bytes_before=int(value["managed_bytes_before"]),
            selected_bytes=int(value["selected_bytes"]),
            plan_checksum=str(value["plan_checksum"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class PruningReport:
    schema_version: str
    plan_checksum: str
    current_bundle_id: str
    current_step: int
    retained_bundle_ids: tuple[str, ...]
    retained_steps: tuple[int, ...]
    pruned_checkpoint_steps: tuple[int, ...]
    deleted_paths: tuple[str, ...]
    already_missing_paths: tuple[str, ...]
    selected_bytes: int
    cumulative_reclaimed_bytes: int
    newly_reclaimed_bytes: int
    managed_bytes_before: int
    managed_bytes_after: int
    current_pointer_unchanged: bool
    retained_bundle_ids_verified: tuple[str, ...]
    recovery: BundleRecoveryReport
    operation_journal_path: str
    idempotent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_checksum": self.plan_checksum,
            "current_bundle_id": self.current_bundle_id,
            "current_step": self.current_step,
            "retained_bundle_ids": list(self.retained_bundle_ids),
            "retained_steps": list(self.retained_steps),
            "pruned_checkpoint_steps": list(self.pruned_checkpoint_steps),
            "deleted_paths": list(self.deleted_paths),
            "already_missing_paths": list(self.already_missing_paths),
            "selected_bytes": self.selected_bytes,
            "cumulative_reclaimed_bytes": self.cumulative_reclaimed_bytes,
            "newly_reclaimed_bytes": self.newly_reclaimed_bytes,
            "managed_bytes_before": self.managed_bytes_before,
            "managed_bytes_after": self.managed_bytes_after,
            "current_pointer_unchanged": self.current_pointer_unchanged,
            "retained_bundle_ids_verified": list(self.retained_bundle_ids_verified),
            "recovery": asdict(self.recovery),
            "operation_journal_path": self.operation_journal_path,
            "idempotent": self.idempotent,
        }


@dataclass(frozen=True)
class _PruningOperation:
    schema_version: str
    plan_checksum: str
    current_bundle_id: str
    state: str
    completed_paths: tuple[str, ...]
    already_missing_paths: tuple[str, ...]
    cumulative_reclaimed_bytes: int
    managed_bytes_after: int | None
    operation_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_checksum": self.plan_checksum,
            "current_bundle_id": self.current_bundle_id,
            "state": self.state,
            "completed_paths": list(self.completed_paths),
            "already_missing_paths": list(self.already_missing_paths),
            "cumulative_reclaimed_bytes": self.cumulative_reclaimed_bytes,
            "managed_bytes_after": self.managed_bytes_after,
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_checksum(self) -> _PruningOperation:
        return replace(self, operation_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.schema_version != PRUNING_OPERATION_SCHEMA_VERSION:
            raise IntegrityError(
                f"unsupported pruning operation schema: {self.schema_version}"
            )
        if self.state not in {"prepared", "deleting", "completed"}:
            raise IntegrityError(f"invalid pruning operation state: {self.state}")
        if self.operation_checksum != self.compute_checksum():
            raise IntegrityError("pruning operation checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["operation_checksum"] = self.operation_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _PruningOperation:
        result = cls(
            schema_version=str(value["schema_version"]),
            plan_checksum=str(value["plan_checksum"]),
            current_bundle_id=str(value["current_bundle_id"]),
            state=str(value["state"]),
            completed_paths=tuple(str(item) for item in value["completed_paths"]),
            already_missing_paths=tuple(
                str(item) for item in value["already_missing_paths"]
            ),
            cumulative_reclaimed_bytes=int(value["cumulative_reclaimed_bytes"]),
            managed_bytes_after=(
                None
                if value.get("managed_bytes_after") is None
                else int(value["managed_bytes_after"])
            ),
            operation_checksum=str(value["operation_checksum"]),
        )
        result.validate()
        return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"expected a JSON object in {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(
    path: Path,
    value: dict[str, Any],
    *,
    before_rename: Callable[[], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = canonical_json_bytes(value) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if before_rename is not None:
        before_rename()
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise IntegrityError(f"unsafe relative path: {value}")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise IntegrityError(f"unsafe relative path: {value}")
    return normalized


def _is_ancestor(parent: str, child: str) -> bool:
    return child.startswith(parent + "/")


def _validate_deletion_path(value: str) -> None:
    normalized = _normalize_relative(value)
    parts = PurePosixPath(normalized).parts
    if parts[0] in {"candidates", "work"} and len(parts) >= 2:
        return
    if parts[0] == "manifests" and len(parts) == 2:
        return
    if len(parts) == 1 and parts[0].startswith(".") and parts[0].endswith(".tmp"):
        return
    if (
        parts[0] == "metrics"
        and len(parts) == 2
        and parts[1].startswith(".")
        and parts[1].endswith(".tmp")
    ):
        return
    raise IntegrityError(f"pruning cannot delete protected path: {value}")


def _path_for(root: Path, relative: str) -> Path:
    normalized = _normalize_relative(relative)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    resolved_root = root.resolve()
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise IntegrityError(f"pruning path escapes the bundle root: {relative}") from exc
    return candidate


def _path_inventory(root: Path, relative: str, category: str) -> PruningPathRecord:
    path = _path_for(root, relative)
    if not os.path.lexists(path):
        raise FileNotFoundError(path)
    entries: list[dict[str, Any]] = []
    byte_count = 0
    file_count = 0
    items = [path]
    if path.is_dir() and not path.is_symlink():
        items.extend(sorted(path.rglob("*"), key=lambda item: item.as_posix()))
    for item in items:
        if item.is_symlink():
            raise IntegrityError(f"pruning refuses symbolic links: {item}")
        item_name = "." if item == path else item.relative_to(path).as_posix()
        if item.is_dir():
            entries.append({"kind": "directory", "path": item_name})
        elif item.is_file():
            size = item.stat().st_size
            entries.append(
                {
                    "kind": "file",
                    "path": item_name,
                    "bytes": size,
                    "sha256": _sha256_file(item),
                }
            )
            byte_count += size
            file_count += 1
        else:
            raise IntegrityError(f"unsupported filesystem object: {item}")
    return PruningPathRecord(
        path=relative,
        category=category,
        byte_count=byte_count,
        file_count=file_count,
        content_checksum=sha256_hex(canonical_json_bytes(entries)),
    )


def _managed_size(root: Path) -> int:
    total = 0
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        if relative.parts and relative.parts[0] == "pruning":
            continue
        if item.is_symlink():
            raise IntegrityError(f"bundle root contains a symbolic link: {item}")
        if item.is_file():
            total += item.stat().st_size
    return total


def _policy(
    config: ExperimentConfig,
    keep_previous: int | None,
    milestone_interval: int | None,
) -> RetentionConfig:
    return RetentionConfig(
        keep_previous=(
            config.retention.keep_previous if keep_previous is None else keep_previous
        ),
        milestone_interval=(
            config.retention.milestone_interval
            if milestone_interval is None
            else milestone_interval
        ),
    )


def _validate_config(root: Path, config: ExperimentConfig) -> None:
    metadata = _load_training_metadata(root)
    expected = config_digest(config)
    if metadata.config_digest != expected:
        raise ResumeConfigurationError(
            "training metadata does not match the pruning configuration: config_digest"
        )


def _lineage_manifests(bundle_store: StepBundleStore) -> tuple[StepBundleManifest, ...]:
    entries = _lineage(bundle_store)
    return tuple(bundle_store.load_manifest(item.bundle_id) for item in entries)


def _retained_manifests(
    manifests: tuple[StepBundleManifest, ...],
    policy: RetentionConfig,
) -> tuple[StepBundleManifest, ...]:
    current_step = manifests[-1].committed_step
    retained_steps = {
        step
        for step in range(
            max(0, current_step - policy.keep_previous),
            current_step + 1,
        )
    }
    if policy.milestone_interval > 0:
        retained_steps.update(
            step
            for step in range(current_step + 1)
            if step % policy.milestone_interval == 0
        )
    return tuple(item for item in manifests if item.committed_step in retained_steps)


def _manifest_store_paths(manifest: StepBundleManifest) -> tuple[str, ...]:
    paths = [manifest.parameter_store.path, manifest.optimizer_store.path]
    if manifest.gradient_store is not None:
        paths.append(manifest.gradient_store.path)
    return tuple(_normalize_relative(item) for item in paths)


def _collect_unprotected(
    root: Path,
    relative: str,
    protected: set[str],
    output: list[str],
) -> None:
    if relative in protected:
        return
    descendants = [item for item in protected if _is_ancestor(relative, item)]
    if not descendants:
        output.append(relative)
        return
    path = _path_for(root, relative)
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError(f"protected store path crosses a non-directory: {relative}")
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        _collect_unprotected(
            root,
            f"{relative}/{child.name}",
            protected,
            output,
        )


def _minimal_paths(paths: set[str]) -> tuple[str, ...]:
    result: list[str] = []
    for path in sorted(paths, key=lambda item: (len(PurePosixPath(item).parts), item)):
        if any(existing == path or _is_ancestor(existing, path) for existing in result):
            continue
        result.append(path)
    return tuple(sorted(result))


def _category(path: str) -> str:
    first = PurePosixPath(path).parts[0]
    if first == "candidates":
        return "candidate_state"
    if first == "work":
        return "work_state"
    if first == "manifests":
        return "unpublished_bundle_manifest"
    return "temporary_metadata"


def build_pruning_plan(
    config: ExperimentConfig,
    *,
    bundle_store_path: str | Path,
    keep_previous: int | None = None,
    milestone_interval: int | None = None,
) -> PruningPlan:
    """Build a deterministic, non-mutating checkpoint-pruning plan."""

    root = Path(bundle_store_path)
    _validate_config(root, config)
    bundle_store = StepBundleStore.open(root)
    current = bundle_store.current_manifest()
    bundle_store.verify(current.bundle_id)
    manifests = _lineage_manifests(bundle_store)
    policy = _policy(config, keep_previous, milestone_interval)
    retained = _retained_manifests(manifests, policy)
    for manifest in retained:
        bundle_store.verify(manifest.bundle_id)

    protected = {
        path for manifest in retained for path in _manifest_store_paths(manifest)
    }
    deletion_candidates: list[str] = []
    for directory_name in ("candidates", "work"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            _collect_unprotected(
                root,
                f"{directory_name}/{child.name}",
                protected,
                deletion_candidates,
            )

    lineage_ids = {item.bundle_id for item in manifests}
    manifest_directory = root / "manifests"
    for path in sorted(manifest_directory.glob("*.json")):
        if path.stem not in lineage_ids:
            deletion_candidates.append(path.relative_to(root).as_posix())

    for path in sorted(root.rglob("*.tmp"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {"candidates", "work", "pruning"}:
            continue
        deletion_candidates.append(relative.as_posix())

    relative_paths = _minimal_paths(set(deletion_candidates))
    records = tuple(
        _path_inventory(root, path, _category(path)) for path in relative_paths
    )
    retained_ids = {item.bundle_id for item in retained}
    pruned = tuple(item for item in manifests if item.bundle_id not in retained_ids)
    plan = PruningPlan(
        schema_version=PRUNING_PLAN_SCHEMA_VERSION,
        current_bundle_id=current.bundle_id,
        current_bundle_checksum=current.bundle_checksum,
        current_pointer_checksum=_sha256_file(root / "CURRENT"),
        current_step=current.committed_step,
        keep_previous=policy.keep_previous,
        milestone_interval=policy.milestone_interval,
        lineage_bundle_ids=tuple(item.bundle_id for item in manifests),
        lineage_steps=tuple(item.committed_step for item in manifests),
        retained_bundle_ids=tuple(item.bundle_id for item in retained),
        retained_steps=tuple(item.committed_step for item in retained),
        retained_store_paths=tuple(sorted(protected)),
        pruned_checkpoint_bundle_ids=tuple(item.bundle_id for item in pruned),
        pruned_checkpoint_steps=tuple(item.committed_step for item in pruned),
        deletion_paths=records,
        managed_bytes_before=_managed_size(root),
        selected_bytes=sum(item.byte_count for item in records),
    ).with_checksum()
    plan.validate()
    return plan


def write_pruning_plan(path: str | Path, plan: PruningPlan) -> None:
    plan.validate()
    _write_json_atomic(Path(path), plan.to_dict())


def load_pruning_plan(path: str | Path) -> PruningPlan:
    return PruningPlan.from_dict(_read_json(Path(path)))


def write_pruning_report(path: str | Path, report: PruningReport) -> None:
    _write_json_atomic(Path(path), report.to_dict())


def _operation_path(root: Path, plan: PruningPlan) -> Path:
    return root / "pruning" / "operations" / f"{plan.plan_checksum}.json"


def _lock_path(root: Path) -> Path:
    return root / "pruning" / "LOCK"


def assert_pruning_inactive(root: str | Path) -> None:
    path = _lock_path(Path(root))
    if path.exists():
        raise PruningInProgressError(f"checkpoint pruning is active for {Path(root)}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _PruningLock:
    def __init__(self, root: Path, plan_checksum: str) -> None:
        self.root = root
        self.path = _lock_path(root)
        self.plan_checksum = plan_checksum
        self.owned = False

    def __enter__(self) -> _PruningLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            value = _read_json(self.path)
            owner_pid = int(value.get("pid", -1))
            if _pid_alive(owner_pid):
                raise PruningInProgressError(
                    f"checkpoint pruning lock is owned by process {owner_pid}"
                )
            self.path.unlink()
            _fsync_directory(self.path.parent)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            data = canonical_json_bytes(
                {
                    "pid": os.getpid(),
                    "plan_checksum": self.plan_checksum,
                }
            ) + b"\n"
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.path.parent)
        self.owned = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self.owned and self.path.exists():
            self.path.unlink()
            _fsync_directory(self.path.parent)
        self.owned = False


def _load_operation(path: Path) -> _PruningOperation:
    return _PruningOperation.from_dict(_read_json(path))


def _write_operation(
    path: Path,
    operation: _PruningOperation,
    *,
    before_rename: Callable[[], None] | None = None,
) -> None:
    value = operation.with_checksum()
    _write_json_atomic(path, value.to_dict(), before_rename=before_rename)


def _verify_current(root: Path, plan: PruningPlan) -> StepBundleStore:
    bundle_store = StepBundleStore.open(root)
    current = bundle_store.current_manifest()
    if current.bundle_id != plan.current_bundle_id:
        raise IntegrityError("CURRENT changed after the pruning plan was created")
    if current.bundle_checksum != plan.current_bundle_checksum:
        raise IntegrityError("CURRENT checksum changed after the pruning plan was created")
    if current.committed_step != plan.current_step:
        raise IntegrityError("CURRENT step changed after the pruning plan was created")
    if _sha256_file(root / "CURRENT") != plan.current_pointer_checksum:
        raise IntegrityError("CURRENT pointer bytes changed after planning")
    return bundle_store


def _verify_retained(
    bundle_store: StepBundleStore,
    plan: PruningPlan,
) -> tuple[str, ...]:
    verified: list[str] = []
    for bundle_id in plan.retained_bundle_ids:
        bundle_store.verify(bundle_id)
        verified.append(bundle_id)
    return tuple(verified)


def _delete_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        raise IntegrityError(f"unsupported deletion target: {path}")
    _fsync_directory(path.parent)


def _report(
    plan: PruningPlan,
    operation: _PruningOperation,
    *,
    root: Path,
    verified: tuple[str, ...],
    recovery: BundleRecoveryReport,
    newly_reclaimed_bytes: int,
    idempotent: bool,
) -> PruningReport:
    managed_after = operation.managed_bytes_after
    if managed_after is None:
        managed_after = _managed_size(root)
    return PruningReport(
        schema_version=PRUNING_REPORT_SCHEMA_VERSION,
        plan_checksum=plan.plan_checksum,
        current_bundle_id=plan.current_bundle_id,
        current_step=plan.current_step,
        retained_bundle_ids=plan.retained_bundle_ids,
        retained_steps=plan.retained_steps,
        pruned_checkpoint_steps=plan.pruned_checkpoint_steps,
        deleted_paths=operation.completed_paths,
        already_missing_paths=operation.already_missing_paths,
        selected_bytes=plan.selected_bytes,
        cumulative_reclaimed_bytes=operation.cumulative_reclaimed_bytes,
        newly_reclaimed_bytes=newly_reclaimed_bytes,
        managed_bytes_before=plan.managed_bytes_before,
        managed_bytes_after=managed_after,
        current_pointer_unchanged=(
            _sha256_file(root / "CURRENT") == plan.current_pointer_checksum
        ),
        retained_bundle_ids_verified=verified,
        recovery=recovery,
        operation_journal_path=_operation_path(root, plan).relative_to(root).as_posix(),
        idempotent=idempotent,
    )


def apply_pruning_plan(
    config: ExperimentConfig,
    *,
    bundle_store_path: str | Path,
    plan: PruningPlan,
    failure_injector: PruningFailureInjector | None = None,
) -> PruningReport:
    """Apply one verified plan while keeping CURRENT byte-identical."""

    plan.validate()
    root = Path(bundle_store_path)
    _validate_config(root, config)
    operation_path = _operation_path(root, plan)
    with _PruningLock(root, plan.plan_checksum):
        bundle_store = _verify_current(root, plan)
        verified = _verify_retained(bundle_store, plan)
        if operation_path.exists():
            operation = _load_operation(operation_path)
            if operation.plan_checksum != plan.plan_checksum:
                raise IntegrityError("pruning journal references a different plan")
            if operation.current_bundle_id != plan.current_bundle_id:
                raise IntegrityError("pruning journal references a different CURRENT")
            if operation.state == "completed":
                recovery = bundle_store.recover()
                return _report(
                    plan,
                    operation,
                    root=root,
                    verified=verified,
                    recovery=recovery,
                    newly_reclaimed_bytes=0,
                    idempotent=True,
                )
        else:
            rebuilt = build_pruning_plan(
                config,
                bundle_store_path=root,
                keep_previous=plan.keep_previous,
                milestone_interval=plan.milestone_interval,
            )
            if rebuilt.to_dict() != plan.to_dict():
                raise IntegrityError("filesystem state no longer matches the pruning plan")
            operation = _PruningOperation(
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

            _write_operation(
                operation_path,
                operation,
                before_rename=before_journal_rename,
            )
            operation = _load_operation(operation_path)

        operation = replace(operation, state="deleting")
        _write_operation(operation_path, operation)
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
            _verify_current(root, plan)
            target = _path_for(root, record.path)
            if not os.path.lexists(target):
                completed.append(record.path)
                missing.append(record.path)
                reclaimed += record.byte_count
                newly_reclaimed += record.byte_count
            else:
                current_inventory = _path_inventory(
                    root,
                    record.path,
                    record.category,
                )
                if current_inventory != record:
                    raise IntegrityError(
                        f"pruning target changed after planning: {record.path}"
                    )
                _delete_path(target)
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
            _write_operation(operation_path, operation)

        bundle_store = _verify_current(root, plan)
        verified = _verify_retained(bundle_store, plan)
        recovery = bundle_store.recover()
        managed_after = _managed_size(root)
        operation = replace(
            operation,
            state="completed",
            managed_bytes_after=managed_after,
        )
        _write_operation(operation_path, operation)
        completed_operation = _load_operation(operation_path)
        report = _report(
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
