"""Atomic publication of backend-neutral bounded training-step bundles."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from .storage import IntegrityError, VersionedTensorStore
from .storage.schema import canonical_json_bytes, sha256_hex

STEP_BUNDLE_SCHEMA_VERSION = "microcolossus.step-bundle.v1"


class BundleFailurePoint(StrEnum):
    BEFORE_MANIFEST_RENAME = "before_manifest_rename"
    BEFORE_CURRENT_RENAME = "before_current_rename"


class BundleSimulatedCrash(RuntimeError):
    """Raised by tests to leave an intentionally unpublished candidate bundle."""


BundleFailureInjector = Callable[[BundleFailurePoint, dict[str, Any]], None]


@dataclass(frozen=True)
class StepStoreReference:
    path: str
    manifest_id: str
    manifest_checksum: str
    tensor_count: int
    chunk_count: int
    logical_bytes: int
    physical_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StepStoreReference:
        return cls(
            path=str(value["path"]),
            manifest_id=str(value["manifest_id"]),
            manifest_checksum=str(value["manifest_checksum"]),
            tensor_count=int(value["tensor_count"]),
            chunk_count=int(value["chunk_count"]),
            logical_bytes=int(value["logical_bytes"]),
            physical_bytes=int(value["physical_bytes"]),
        )


@dataclass(frozen=True)
class StepBundleManifest:
    schema_version: str
    bundle_id: str
    parent_bundle_id: str | None
    committed_step: int
    created_at_utc: str
    parameter_store: StepStoreReference
    optimizer_store: StepStoreReference
    gradient_store: StepStoreReference | None
    batch_checksum: str
    bundle_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "parent_bundle_id": self.parent_bundle_id,
            "committed_step": self.committed_step,
            "created_at_utc": self.created_at_utc,
            "parameter_store": self.parameter_store.to_dict(),
            "optimizer_store": self.optimizer_store.to_dict(),
            "gradient_store": (
                None if self.gradient_store is None else self.gradient_store.to_dict()
            ),
            "batch_checksum": self.batch_checksum,
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_computed_checksum(self) -> StepBundleManifest:
        return replace(self, bundle_checksum=self.compute_checksum())

    def validate_checksum(self) -> None:
        expected = self.compute_checksum()
        if self.bundle_checksum != expected:
            raise IntegrityError(
                f"step bundle checksum mismatch: {self.bundle_checksum} != {expected}"
            )

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["bundle_checksum"] = self.bundle_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StepBundleManifest:
        gradient_value = value.get("gradient_store")
        return cls(
            schema_version=str(value["schema_version"]),
            bundle_id=str(value["bundle_id"]),
            parent_bundle_id=(
                None
                if value.get("parent_bundle_id") is None
                else str(value["parent_bundle_id"])
            ),
            committed_step=int(value["committed_step"]),
            created_at_utc=str(value["created_at_utc"]),
            parameter_store=StepStoreReference.from_dict(dict(value["parameter_store"])),
            optimizer_store=StepStoreReference.from_dict(dict(value["optimizer_store"])),
            gradient_store=(
                None
                if gradient_value is None
                else StepStoreReference.from_dict(dict(gradient_value))
            ),
            batch_checksum=str(value["batch_checksum"]),
            bundle_checksum=str(value["bundle_checksum"]),
        )


@dataclass(frozen=True)
class BundlePublicationTelemetry:
    manifest_fsync_seconds: float
    manifest_publication_seconds: float
    current_fsync_seconds: float
    current_publication_seconds: float
    metadata_bytes_written: int


@dataclass(frozen=True)
class BundleVerificationReport:
    bundle_id: str
    committed_step: int
    parameter_tensor_count: int
    optimizer_tensor_count: int
    gradient_tensor_count: int


@dataclass(frozen=True)
class BundleRecoveryReport:
    current_bundle_id: str
    unpublished_bundle_ids: tuple[str, ...]
    temporary_paths: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> float:
    started = time.perf_counter()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return time.perf_counter() - started


def _write_file_fsynced(path: Path, data: bytes) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return time.perf_counter() - started


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"expected a JSON object in {path}")
    return value


class StepBundleStore:
    """Atomic root pointer over parameter, optimizer, and gradient tensor stores."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def create(cls, root: str | Path) -> StepBundleStore:
        destination = Path(root)
        if destination.exists():
            raise FileExistsError(f"step bundle store already exists: {destination}")
        destination.mkdir(parents=True)
        (destination / "manifests").mkdir()
        (destination / "work").mkdir()
        (destination / "candidates").mkdir()
        _fsync_directory(destination / "manifests")
        _fsync_directory(destination / "work")
        _fsync_directory(destination / "candidates")
        _fsync_directory(destination)
        return cls(destination)

    @classmethod
    def open(cls, root: str | Path) -> StepBundleStore:
        destination = Path(root)
        if not (destination / "manifests").is_dir():
            raise IntegrityError(f"not a step bundle store: {destination}")
        return cls(destination)

    def _relative_store_path(self, path: str | Path) -> str:
        resolved_root = self.root.resolve()
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError("referenced tensor stores must remain inside the bundle root") from exc

    def _store_reference(self, path: str | Path) -> StepStoreReference:
        relative = self._relative_store_path(path)
        store = VersionedTensorStore.open(self.root / relative)
        manifest = store.current_manifest()
        verification = store.verify(manifest.manifest_id)
        return StepStoreReference(
            path=relative,
            manifest_id=manifest.manifest_id,
            manifest_checksum=manifest.manifest_checksum,
            tensor_count=verification.tensor_count,
            chunk_count=verification.chunk_count,
            logical_bytes=verification.logical_bytes,
            physical_bytes=verification.physical_bytes,
        )

    def _manifest_path(self, bundle_id: str) -> Path:
        return self.root / "manifests" / f"{bundle_id}.json"

    def current_manifest(self) -> StepBundleManifest:
        pointer = _read_json(self.root / "CURRENT")
        bundle_id = str(pointer["bundle_id"])
        checksum = str(pointer["bundle_checksum"])
        manifest = self.load_manifest(bundle_id)
        if manifest.bundle_checksum != checksum:
            raise IntegrityError("CURRENT bundle checksum does not match the manifest")
        return manifest

    def load_manifest(self, bundle_id: str) -> StepBundleManifest:
        manifest = StepBundleManifest.from_dict(_read_json(self._manifest_path(bundle_id)))
        if manifest.schema_version != STEP_BUNDLE_SCHEMA_VERSION:
            raise IntegrityError(f"unsupported step bundle schema: {manifest.schema_version}")
        manifest.validate_checksum()
        return manifest

    def publish(
        self,
        *,
        committed_step: int,
        parameter_store_path: str | Path,
        optimizer_store_path: str | Path,
        gradient_store_path: str | Path | None,
        batch_checksum: str,
        failure_injector: BundleFailureInjector | None = None,
    ) -> tuple[StepBundleManifest, BundlePublicationTelemetry]:
        current_path = self.root / "CURRENT"
        parent = self.current_manifest() if current_path.exists() else None
        if parent is None:
            if committed_step != 0:
                raise ValueError("the first published bundle must be committed step 0")
        elif committed_step != parent.committed_step + 1:
            raise ValueError("bundle committed_step must advance exactly by one")

        manifest = StepBundleManifest(
            schema_version=STEP_BUNDLE_SCHEMA_VERSION,
            bundle_id=f"bundle-{uuid.uuid4().hex}",
            parent_bundle_id=None if parent is None else parent.bundle_id,
            committed_step=committed_step,
            created_at_utc=_utc_now(),
            parameter_store=self._store_reference(parameter_store_path),
            optimizer_store=self._store_reference(optimizer_store_path),
            gradient_store=(
                None if gradient_store_path is None else self._store_reference(gradient_store_path)
            ),
            batch_checksum=batch_checksum,
        ).with_computed_checksum()

        manifest_data = canonical_json_bytes(manifest.to_dict()) + b"\n"
        manifest_path = self._manifest_path(manifest.bundle_id)
        manifest_temporary = manifest_path.with_name(
            f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        manifest_fsync = _write_file_fsynced(manifest_temporary, manifest_data)
        if failure_injector is not None:
            failure_injector(
                BundleFailurePoint.BEFORE_MANIFEST_RENAME,
                {"bundle_id": manifest.bundle_id},
            )
        started = time.perf_counter()
        os.replace(manifest_temporary, manifest_path)
        manifest_publication = time.perf_counter() - started
        manifest_fsync += _fsync_directory(manifest_path.parent)

        pointer_data = canonical_json_bytes(
            {
                "bundle_id": manifest.bundle_id,
                "bundle_checksum": manifest.bundle_checksum,
            }
        ) + b"\n"
        pointer_temporary = self.root / f".CURRENT.{uuid.uuid4().hex}.tmp"
        current_fsync = _write_file_fsynced(pointer_temporary, pointer_data)
        if failure_injector is not None:
            failure_injector(
                BundleFailurePoint.BEFORE_CURRENT_RENAME,
                {"bundle_id": manifest.bundle_id},
            )
        started = time.perf_counter()
        os.replace(pointer_temporary, current_path)
        current_publication = time.perf_counter() - started
        current_fsync += _fsync_directory(self.root)
        return manifest, BundlePublicationTelemetry(
            manifest_fsync_seconds=manifest_fsync,
            manifest_publication_seconds=manifest_publication,
            current_fsync_seconds=current_fsync,
            current_publication_seconds=current_publication,
            metadata_bytes_written=len(manifest_data) + len(pointer_data),
        )

    def verify(self, bundle_id: str | None = None) -> BundleVerificationReport:
        manifest = self.current_manifest() if bundle_id is None else self.load_manifest(bundle_id)
        references = [manifest.parameter_store, manifest.optimizer_store]
        if manifest.gradient_store is not None:
            references.append(manifest.gradient_store)
        counts: list[int] = []
        for reference in references:
            store = VersionedTensorStore.open(self.root / reference.path)
            child_manifest = store.load_manifest(reference.manifest_id)
            if child_manifest.manifest_checksum != reference.manifest_checksum:
                raise IntegrityError(
                    f"child manifest checksum mismatch for {reference.path}"
                )
            report = store.verify(reference.manifest_id)
            if (
                report.tensor_count != reference.tensor_count
                or report.chunk_count != reference.chunk_count
                or report.logical_bytes != reference.logical_bytes
                or report.physical_bytes != reference.physical_bytes
            ):
                raise IntegrityError(f"child store verification mismatch for {reference.path}")
            counts.append(report.tensor_count)
        return BundleVerificationReport(
            bundle_id=manifest.bundle_id,
            committed_step=manifest.committed_step,
            parameter_tensor_count=counts[0],
            optimizer_tensor_count=counts[1],
            gradient_tensor_count=0 if manifest.gradient_store is None else counts[2],
        )

    def recover(self) -> BundleRecoveryReport:
        current = self.current_manifest()
        self.verify(current.bundle_id)
        manifests: dict[str, StepBundleManifest] = {}
        for path in sorted((self.root / "manifests").glob("*.json")):
            try:
                manifest = StepBundleManifest.from_dict(_read_json(path))
                manifest.validate_checksum()
            except (IntegrityError, ValueError):
                continue
            manifests[manifest.bundle_id] = manifest
        reachable: set[str] = set()
        candidate: str | None = current.bundle_id
        while candidate is not None and candidate not in reachable:
            reachable.add(candidate)
            manifest = manifests.get(candidate)
            candidate = None if manifest is None else manifest.parent_bundle_id
        temporary_paths = tuple(
            sorted(
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob("*.tmp")
            )
        )
        return BundleRecoveryReport(
            current_bundle_id=current.bundle_id,
            unpublished_bundle_ids=tuple(sorted(set(manifests) - reachable)),
            temporary_paths=temporary_paths,
        )
