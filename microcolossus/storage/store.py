"""Synchronous, versioned, crash-recoverable tensor storage."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .codec import TensorPayload
from .schema import (
    JOURNAL_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    ChunkRecord,
    FailurePoint,
    JournalEntry,
    Manifest,
    StoreLimits,
    StoreMetadata,
    StoreTelemetry,
    TensorRecord,
    TransactionState,
    canonical_json_bytes,
    sha256_hex,
)

FailureInjector = Callable[[FailurePoint, dict[str, Any]], None]


class TensorStoreError(RuntimeError):
    """Base error for tensor store operations."""


class StoreNotInitializedError(TensorStoreError):
    pass


class IntegrityError(TensorStoreError):
    pass


class BudgetExceededError(TensorStoreError):
    pass


class TransactionStateError(TensorStoreError):
    pass


class SimulatedCrash(TensorStoreError):
    """Raised by tests to leave an intentionally incomplete transaction."""


@dataclass(frozen=True)
class VerificationReport:
    manifest_id: str
    tensor_count: int
    chunk_count: int
    logical_bytes: int
    physical_bytes: int


@dataclass(frozen=True)
class RecoveryReport:
    current_manifest_id: str
    incomplete_transactions: tuple[str, ...]
    aborted_transactions: tuple[str, ...]
    unpublished_manifests: tuple[str, ...]
    orphan_paths: tuple[str, ...]
    actions: tuple[str, ...]
    telemetry: StoreTelemetry


@dataclass(frozen=True)
class CommitResult:
    manifest: Manifest
    telemetry: StoreTelemetry


def utc_now() -> str:
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


def _atomic_write(path: Path, data: bytes) -> tuple[float, float]:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fsync_seconds = _write_file_fsynced(temporary, data)
    started = time.perf_counter()
    os.replace(temporary, path)
    rename_seconds = time.perf_counter() - started
    fsync_seconds += _fsync_directory(path.parent)
    return fsync_seconds, rename_seconds


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"expected a JSON object in {path}")
    return value


def _file_content_size(root: Path) -> int:
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _chunk_path(root: Path, chunk_id: str) -> Path:
    return root / "chunks" / chunk_id[:2] / f"{chunk_id}.chunk"


def _manifest_path(root: Path, manifest_id: str) -> Path:
    return root / "manifests" / f"{manifest_id}.json"


def _journal_path(root: Path, transaction_id: str) -> Path:
    return root / "transactions" / transaction_id / "journal.jsonl"


def _telemetry_path(root: Path) -> Path:
    return root / "telemetry" / "events.jsonl"


def _current_pointer_bytes(manifest: Manifest) -> bytes:
    return canonical_json_bytes(
        {
            "manifest_id": manifest.manifest_id,
            "manifest_checksum": manifest.manifest_checksum,
        }
    ) + b"\n"


class VersionedTensorStore:
    """A backend-neutral tensor store with atomic manifest publication."""

    def __init__(self, root: str | Path, metadata: StoreMetadata) -> None:
        self.root = Path(root)
        self.metadata = metadata

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        limits: StoreLimits | None = None,
    ) -> VersionedTensorStore:
        destination = Path(root)
        if destination.exists():
            raise FileExistsError(f"tensor store already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir()
        try:
            for directory in ("chunks", "manifests", "transactions", "telemetry"):
                (temporary / directory).mkdir()
            metadata = StoreMetadata(
                schema_version=STORE_SCHEMA_VERSION,
                store_id=uuid.uuid4().hex,
                created_at_utc=utc_now(),
                limits=limits or StoreLimits(),
            )
            _write_file_fsynced(
                temporary / "store.json",
                canonical_json_bytes(metadata.to_dict()) + b"\n",
            )
            initial = Manifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                manifest_id=f"manifest-{uuid.uuid4().hex}",
                parent_manifest_id=None,
                committed_step=-1,
                created_at_utc=utc_now(),
                tensors=(),
                chunks=(),
                aggregate_logical_bytes=0,
                aggregate_physical_bytes=0,
            ).with_computed_checksum()
            _write_file_fsynced(
                _manifest_path(temporary, initial.manifest_id),
                canonical_json_bytes(initial.to_dict()) + b"\n",
            )
            _write_file_fsynced(temporary / "CURRENT", _current_pointer_bytes(initial))
            _write_file_fsynced(temporary / "CUMULATIVE_WRITES", b"0\n")
            for directory in (
                temporary / "chunks",
                temporary / "manifests",
                temporary / "transactions",
                temporary / "telemetry",
                temporary,
            ):
                _fsync_directory(directory)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return cls.open(destination)

    @classmethod
    def open(cls, root: str | Path) -> VersionedTensorStore:
        destination = Path(root)
        metadata_path = destination / "store.json"
        if not metadata_path.exists():
            raise StoreNotInitializedError(f"not a tensor store: {destination}")
        metadata = StoreMetadata.from_dict(_read_json(metadata_path))
        if metadata.schema_version != STORE_SCHEMA_VERSION:
            raise IntegrityError(
                f"unsupported store schema: {metadata.schema_version}"
            )
        store = cls(destination, metadata)
        store.current_manifest().validate_checksum()
        return store

    @property
    def limits(self) -> StoreLimits:
        return self.metadata.limits

    def _current_pointer(self) -> dict[str, Any]:
        value = _read_json(self.root / "CURRENT")
        if "manifest_id" not in value or "manifest_checksum" not in value:
            raise IntegrityError("CURRENT pointer is incomplete")
        return value

    def current_manifest_id(self) -> str:
        return str(self._current_pointer()["manifest_id"])

    def load_manifest(self, manifest_id: str) -> Manifest:
        path = _manifest_path(self.root, manifest_id)
        if not path.exists():
            raise IntegrityError(f"manifest does not exist: {manifest_id}")
        manifest = Manifest.from_dict(_read_json(path))
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise IntegrityError(
                f"unsupported manifest schema: {manifest.schema_version}"
            )
        try:
            manifest.validate_structure()
            manifest.validate_checksum()
        except ValueError as exc:
            raise IntegrityError(str(exc)) from exc
        return manifest

    def current_manifest(self) -> Manifest:
        pointer = self._current_pointer()
        manifest = self.load_manifest(str(pointer["manifest_id"]))
        if manifest.manifest_checksum != str(pointer["manifest_checksum"]):
            raise IntegrityError("CURRENT manifest checksum does not match manifest")
        return manifest

    def begin_transaction(
        self,
        *,
        committed_step: int,
        failure_injector: FailureInjector | None = None,
    ) -> StoreTransaction:
        current = self.current_manifest()
        if committed_step <= current.committed_step:
            raise ValueError(
                "committed_step must be greater than the current committed step"
            )
        return StoreTransaction(
            store=self,
            parent=current,
            committed_step=committed_step,
            failure_injector=failure_injector,
        )

    def _cumulative_writes(self) -> int:
        try:
            return int((self.root / "CUMULATIVE_WRITES").read_text().strip())
        except (OSError, ValueError) as exc:
            raise IntegrityError("CUMULATIVE_WRITES is invalid") from exc

    def _set_cumulative_writes(self, value: int) -> None:
        _atomic_write(self.root / "CUMULATIVE_WRITES", f"{value}\n".encode())

    def _append_telemetry(self, telemetry: StoreTelemetry) -> None:
        path = _telemetry_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(telemetry.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)

    def _record_telemetry(
        self,
        telemetry: StoreTelemetry,
        *,
        counted_writes: int,
    ) -> StoreTelemetry:
        cumulative = self._cumulative_writes() + counted_writes
        if counted_writes:
            self._set_cumulative_writes(cumulative)
        updated = replace(
            telemetry,
            store_size_bytes=_file_content_size(self.root),
            cumulative_physical_bytes_written=cumulative,
        )
        self._append_telemetry(updated)
        return updated

    def verify(self, manifest_id: str | None = None) -> VerificationReport:
        manifest = (
            self.current_manifest() if manifest_id is None else self.load_manifest(manifest_id)
        )
        chunk_map = {record.chunk_id: record for record in manifest.chunks}
        for chunk in manifest.chunks:
            path = self.root / chunk.storage_path
            if not path.exists():
                raise IntegrityError(f"missing chunk: {chunk.chunk_id}")
            data = path.read_bytes()
            if len(data) != chunk.byte_length:
                raise IntegrityError(f"chunk length mismatch: {chunk.chunk_id}")
            if sha256_hex(data) != chunk.checksum:
                raise IntegrityError(f"chunk checksum mismatch: {chunk.chunk_id}")
        for tensor in manifest.tensors:
            data = b"".join(
                (self.root / chunk_map[chunk_id].storage_path).read_bytes()
                for chunk_id in tensor.chunk_ids
            )
            if len(data) != tensor.byte_length:
                raise IntegrityError(f"tensor length mismatch: {tensor.tensor_id}")
            if sha256_hex(data) != tensor.checksum:
                raise IntegrityError(f"tensor checksum mismatch: {tensor.tensor_id}")
        return VerificationReport(
            manifest_id=manifest.manifest_id,
            tensor_count=len(manifest.tensors),
            chunk_count=len(manifest.chunks),
            logical_bytes=manifest.aggregate_logical_bytes,
            physical_bytes=manifest.aggregate_physical_bytes,
        )

    def _read_tensor_record(
        self, tensor_id: str, manifest_id: str | None
    ) -> tuple[Manifest, TensorRecord]:
        manifest = (
            self.current_manifest() if manifest_id is None else self.load_manifest(manifest_id)
        )
        for record in manifest.tensors:
            if record.tensor_id == tensor_id:
                return manifest, record
        raise KeyError(f"tensor not found: {tensor_id}")

    def read_chunk(self, chunk_id: str, manifest_id: str | None = None) -> bytes:
        started = time.perf_counter()
        manifest = (
            self.current_manifest() if manifest_id is None else self.load_manifest(manifest_id)
        )
        record = next(
            (item for item in manifest.chunks if item.chunk_id == chunk_id), None
        )
        if record is None:
            raise KeyError(f"chunk not found: {chunk_id}")
        data = (self.root / record.storage_path).read_bytes()
        checksum_started = time.perf_counter()
        checksum = sha256_hex(data)
        checksum_seconds = time.perf_counter() - checksum_started
        if checksum != record.checksum or len(data) != record.byte_length:
            raise IntegrityError(f"chunk integrity failure: {chunk_id}")
        telemetry = StoreTelemetry(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            operation="read_chunk",
            transaction_id=None,
            logical_bytes_read=len(data),
            physical_bytes_read=len(data),
            chunk_reads=1,
            checksum_seconds=checksum_seconds,
            read_seconds=time.perf_counter() - started,
        )
        self._record_telemetry(telemetry, counted_writes=0)
        return data

    def read_tensor(
        self, tensor_id: str, manifest_id: str | None = None
    ) -> TensorPayload:
        started = time.perf_counter()
        manifest, tensor = self._read_tensor_record(tensor_id, manifest_id)
        chunk_map = {item.chunk_id: item for item in manifest.chunks}
        parts: list[bytes] = []
        physical = 0
        for chunk_id in tensor.chunk_ids:
            chunk = chunk_map[chunk_id]
            data = (self.root / chunk.storage_path).read_bytes()
            if len(data) != chunk.byte_length or sha256_hex(data) != chunk.checksum:
                raise IntegrityError(f"chunk integrity failure: {chunk_id}")
            physical += len(data)
            parts.append(data)
        data = b"".join(parts)
        checksum_started = time.perf_counter()
        checksum = sha256_hex(data)
        checksum_seconds = time.perf_counter() - checksum_started
        if len(data) != tensor.byte_length or checksum != tensor.checksum:
            raise IntegrityError(f"tensor integrity failure: {tensor_id}")
        payload = TensorPayload(
            logical_name=tensor.logical_name,
            kind=tensor.kind,
            shape=tensor.shape,
            dtype=tensor.dtype,
            byte_order=tensor.byte_order,
            data=data,
            metadata=tensor.metadata,
        )
        telemetry = StoreTelemetry(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            operation="read_tensor",
            transaction_id=None,
            logical_bytes_read=len(data),
            physical_bytes_read=physical,
            chunk_reads=len(tensor.chunk_ids),
            checksum_seconds=checksum_seconds,
            read_seconds=time.perf_counter() - started,
        )
        self._record_telemetry(telemetry, counted_writes=0)
        return payload

    def read_tensor_range(
        self,
        tensor_id: str,
        *,
        start: int,
        length: int,
        manifest_id: str | None = None,
    ) -> bytes:
        if start < 0 or length < 0:
            raise ValueError("start and length cannot be negative")
        started = time.perf_counter()
        manifest, tensor = self._read_tensor_record(tensor_id, manifest_id)
        end = start + length
        if end > tensor.byte_length:
            raise ValueError("requested range exceeds tensor byte length")
        if length == 0:
            return b""
        chunk_map = {item.chunk_id: item for item in manifest.chunks}
        position = 0
        selected: list[bytes] = []
        physical = 0
        reads = 0
        checksum_seconds = 0.0
        for chunk_id in tensor.chunk_ids:
            chunk = chunk_map[chunk_id]
            chunk_start = position
            chunk_end = position + chunk.byte_length
            position = chunk_end
            overlap_start = max(start, chunk_start)
            overlap_end = min(end, chunk_end)
            if overlap_start >= overlap_end:
                continue
            data = (self.root / chunk.storage_path).read_bytes()
            checksum_started = time.perf_counter()
            checksum = sha256_hex(data)
            checksum_seconds += time.perf_counter() - checksum_started
            if len(data) != chunk.byte_length or checksum != chunk.checksum:
                raise IntegrityError(f"chunk integrity failure: {chunk_id}")
            selected.append(
                data[overlap_start - chunk_start : overlap_end - chunk_start]
            )
            physical += len(data)
            reads += 1
        result = b"".join(selected)
        if len(result) != length:
            raise IntegrityError("tensor range reconstruction produced the wrong length")
        telemetry = StoreTelemetry(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            operation="read_tensor_range",
            transaction_id=None,
            logical_bytes_read=len(result),
            physical_bytes_read=physical,
            chunk_reads=reads,
            checksum_seconds=checksum_seconds,
            read_seconds=time.perf_counter() - started,
        )
        self._record_telemetry(telemetry, counted_writes=0)
        return result

    def _all_manifests(self) -> dict[str, Manifest]:
        manifests: dict[str, Manifest] = {}
        for path in sorted((self.root / "manifests").glob("*.json")):
            try:
                manifest = Manifest.from_dict(_read_json(path))
                manifest.validate_checksum()
            except (IntegrityError, ValueError):
                continue
            manifests[manifest.manifest_id] = manifest
        return manifests

    def _reachable_manifest_ids(
        self, current_id: str, manifests: dict[str, Manifest]
    ) -> set[str]:
        reachable: set[str] = set()
        candidate: str | None = current_id
        while candidate is not None and candidate not in reachable:
            reachable.add(candidate)
            manifest = manifests.get(candidate)
            candidate = manifest.parent_manifest_id if manifest is not None else None
        return reachable

    def recover(self) -> RecoveryReport:
        started = time.perf_counter()
        current = self.current_manifest()
        self.verify(current.manifest_id)
        incomplete: list[str] = []
        aborted: list[str] = []
        actions: list[str] = []
        for transaction_dir in sorted((self.root / "transactions").iterdir()):
            if not transaction_dir.is_dir():
                continue
            entries = _read_journal(transaction_dir / "journal.jsonl")
            if not entries:
                continue
            last = entries[-1]
            if last.state in {TransactionState.COMMITTED, TransactionState.ABORTED}:
                continue
            incomplete.append(last.transaction_id)
            _append_journal_entry(
                self.root,
                transaction_id=last.transaction_id,
                state=TransactionState.ABORTED,
                parent_manifest_id=last.parent_manifest_id,
                candidate_manifest_id=last.candidate_manifest_id,
                intended_chunk_ids=last.intended_chunk_ids,
                written_chunk_ids=last.written_chunk_ids,
                message="recovery kept the last committed manifest authoritative",
            )
            aborted.append(last.transaction_id)
            actions.append(f"aborted incomplete transaction {last.transaction_id}")

        manifests = self._all_manifests()
        reachable = self._reachable_manifest_ids(current.manifest_id, manifests)
        unpublished = tuple(sorted(set(manifests) - reachable))
        for manifest_id in unpublished:
            actions.append(f"reported unpublished manifest {manifest_id}")

        referenced_paths = {
            chunk.storage_path
            for manifest in manifests.values()
            for chunk in manifest.chunks
        }
        orphan_paths: list[str] = []
        for path in (self.root / "chunks").rglob("*.chunk"):
            relative = path.relative_to(self.root).as_posix()
            if relative not in referenced_paths:
                orphan_paths.append(relative)
        for path in (self.root / "transactions").rglob("*.part"):
            orphan_paths.append(path.relative_to(self.root).as_posix())
        for path in self.root.rglob("*.tmp"):
            orphan_paths.append(path.relative_to(self.root).as_posix())
        for path in sorted(set(orphan_paths)):
            actions.append(f"reported orphan path {path}")

        telemetry = StoreTelemetry(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            operation="recover",
            transaction_id=None,
            recovery_seconds=time.perf_counter() - started,
            actions=tuple(actions),
        )
        telemetry = self._record_telemetry(telemetry, counted_writes=0)
        return RecoveryReport(
            current_manifest_id=current.manifest_id,
            incomplete_transactions=tuple(incomplete),
            aborted_transactions=tuple(aborted),
            unpublished_manifests=unpublished,
            orphan_paths=tuple(sorted(set(orphan_paths))),
            actions=tuple(actions),
            telemetry=telemetry,
        )


class StoreTransaction:
    """A synchronous copy-on-write transaction."""

    def __init__(
        self,
        *,
        store: VersionedTensorStore,
        parent: Manifest,
        committed_step: int,
        failure_injector: FailureInjector | None,
    ) -> None:
        self.store = store
        self.parent = parent
        self.committed_step = committed_step
        self.failure_injector = failure_injector
        self.transaction_id = f"tx-{uuid.uuid4().hex}"
        self.candidate_manifest_id = f"manifest-{uuid.uuid4().hex}"
        self._payloads: dict[str, TensorPayload] = {}
        self._explicit_versions: dict[str, int] = {}
        self._state = TransactionState.PREPARED
        self._written_chunk_ids: list[str] = []
        (self.store.root / "transactions" / self.transaction_id).mkdir(parents=True)
        _append_journal_entry(
            self.store.root,
            transaction_id=self.transaction_id,
            state=self._state,
            parent_manifest_id=self.parent.manifest_id,
            candidate_manifest_id=self.candidate_manifest_id,
            intended_chunk_ids=(),
            written_chunk_ids=(),
        )

    @property
    def state(self) -> TransactionState:
        return self._state

    def put_tensor(
        self,
        payload: TensorPayload,
        *,
        tensor_id: str | None = None,
        version: int | None = None,
    ) -> str:
        if self._state is not TransactionState.PREPARED:
            raise TransactionStateError("tensors can only be staged while prepared")
        identifier = tensor_id or payload.stable_tensor_id()
        if not identifier:
            raise ValueError("tensor_id cannot be empty")
        self._payloads[identifier] = payload
        if version is not None:
            if version < 0:
                raise ValueError("version cannot be negative")
            self._explicit_versions[identifier] = version
        return identifier

    def put_many(self, payloads: Iterable[TensorPayload]) -> tuple[str, ...]:
        return tuple(self.put_tensor(payload) for payload in payloads)

    def abort(self, message: str = "transaction aborted") -> None:
        if self._state in {TransactionState.COMMITTED, TransactionState.ABORTED}:
            return
        self._state = TransactionState.ABORTED
        _append_journal_entry(
            self.store.root,
            transaction_id=self.transaction_id,
            state=self._state,
            parent_manifest_id=self.parent.manifest_id,
            candidate_manifest_id=self.candidate_manifest_id,
            intended_chunk_ids=(),
            written_chunk_ids=tuple(self._written_chunk_ids),
            message=message,
        )
        self._payloads.clear()
        self._explicit_versions.clear()

    def _inject(self, point: FailurePoint, **context: Any) -> None:
        if self.failure_injector is not None:
            self.failure_injector(point, context)

    def _build_candidate(
        self,
    ) -> tuple[Manifest, dict[str, bytes], int, int]:
        parent_tensors = {item.tensor_id: item for item in self.parent.tensors}
        tensor_records = dict(parent_tensors)
        parent_chunks = {item.chunk_id: item for item in self.parent.chunks}
        all_chunks = dict(parent_chunks)
        new_chunk_data: dict[str, bytes] = {}
        chunks_reused = 0
        for tensor_id, payload in sorted(self._payloads.items()):
            chunk_ids: list[str] = []
            for start in range(0, payload.byte_length, self.store.limits.chunk_size_bytes):
                data = payload.data[start : start + self.store.limits.chunk_size_bytes]
                chunk_id = sha256_hex(data)
                chunk_ids.append(chunk_id)
                path = _chunk_path(self.store.root, chunk_id)
                relative = path.relative_to(self.store.root).as_posix()
                if chunk_id in all_chunks or path.exists():
                    chunks_reused += 1
                else:
                    new_chunk_data[chunk_id] = data
                all_chunks.setdefault(
                    chunk_id,
                    ChunkRecord(
                        chunk_id=chunk_id,
                        storage_path=relative,
                        byte_offset=0,
                        byte_length=len(data),
                        checksum=chunk_id,
                        compression="none",
                        creating_transaction=self.transaction_id,
                    ),
                )
            prior = parent_tensors.get(tensor_id)
            version = self._explicit_versions.get(
                tensor_id, prior.version + 1 if prior is not None else 0
            )
            tensor_records[tensor_id] = TensorRecord(
                tensor_id=tensor_id,
                logical_name=payload.logical_name,
                kind=payload.kind,
                shape=payload.shape,
                dtype=payload.dtype,
                byte_order=payload.byte_order,
                version=version,
                chunk_ids=tuple(chunk_ids),
                byte_length=payload.byte_length,
                checksum=payload.checksum,
                committed_step=self.committed_step,
                metadata=payload.metadata,
            )
        referenced = {
            chunk_id
            for tensor in tensor_records.values()
            for chunk_id in tensor.chunk_ids
        }
        manifest_chunks = tuple(
            sorted(
                (record for chunk_id, record in all_chunks.items() if chunk_id in referenced),
                key=lambda item: item.chunk_id,
            )
        )
        manifest_tensors = tuple(sorted(tensor_records.values(), key=lambda item: item.tensor_id))
        candidate = Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            manifest_id=self.candidate_manifest_id,
            parent_manifest_id=self.parent.manifest_id,
            committed_step=self.committed_step,
            created_at_utc=utc_now(),
            tensors=manifest_tensors,
            chunks=manifest_chunks,
            aggregate_logical_bytes=sum(item.byte_length for item in manifest_tensors),
            aggregate_physical_bytes=sum(item.byte_length for item in manifest_chunks),
        ).with_computed_checksum()
        return candidate, new_chunk_data, chunks_reused, sum(
            payload.byte_length for payload in self._payloads.values()
        )

    def _check_budgets(self, candidate: Manifest, new_chunk_data: dict[str, bytes]) -> None:
        for chunk_id, data in new_chunk_data.items():
            if len(data) > self.store.limits.max_staging_bytes:
                raise BudgetExceededError(
                    f"chunk {chunk_id} exceeds the staging budget"
                )
        existing_content = _file_content_size(self.store.root)
        manifest_bytes = len(canonical_json_bytes(candidate.to_dict()) + b"\n")
        pointer_bytes = len(_current_pointer_bytes(candidate))
        new_bytes = sum(len(data) for data in new_chunk_data.values())
        conservative_journal_allowance = 16 * 1024
        projected = (
            existing_content
            + manifest_bytes
            + pointer_bytes
            + new_bytes
            + conservative_journal_allowance
        )
        if projected > self.store.limits.max_storage_bytes:
            raise BudgetExceededError(
                f"projected store content {projected} exceeds "
                f"max_storage_bytes={self.store.limits.max_storage_bytes}"
            )

    def _write_chunk(self, chunk_id: str, data: bytes) -> tuple[float, float]:
        final_path = _chunk_path(self.store.root, chunk_id)
        if final_path.exists():
            existing = final_path.read_bytes()
            if sha256_hex(existing) != chunk_id:
                raise IntegrityError(f"existing chunk is corrupt: {chunk_id}")
            return 0.0, 0.0
        final_path.parent.mkdir(parents=True, exist_ok=True)
        transaction_dir = self.store.root / "transactions" / self.transaction_id
        temporary = transaction_dir / f"{chunk_id}.part"
        self._inject(FailurePoint.BEFORE_CHUNK_WRITE, chunk_id=chunk_id)
        write_started = time.perf_counter()
        with temporary.open("wb") as handle:
            midpoint = len(data) // 2
            handle.write(data[:midpoint])
            handle.flush()
            self._inject(FailurePoint.DURING_CHUNK_WRITE, chunk_id=chunk_id)
            handle.write(data[midpoint:])
            handle.flush()
            self._inject(FailurePoint.BEFORE_CHUNK_FSYNC, chunk_id=chunk_id)
            fsync_started = time.perf_counter()
            os.fsync(handle.fileno())
            fsync_seconds = time.perf_counter() - fsync_started
        write_seconds = time.perf_counter() - write_started
        os.replace(temporary, final_path)
        fsync_seconds += _fsync_directory(final_path.parent)
        self._written_chunk_ids.append(chunk_id)
        return write_seconds, fsync_seconds

    def _validate_candidate(self, candidate: Manifest) -> float:
        started = time.perf_counter()
        chunk_map = {item.chunk_id: item for item in candidate.chunks}
        for chunk in candidate.chunks:
            path = self.store.root / chunk.storage_path
            if not path.exists():
                raise IntegrityError(f"candidate chunk is missing: {chunk.chunk_id}")
            data = path.read_bytes()
            if len(data) != chunk.byte_length or sha256_hex(data) != chunk.checksum:
                raise IntegrityError(f"candidate chunk is corrupt: {chunk.chunk_id}")
        for tensor in candidate.tensors:
            data = b"".join(
                (self.store.root / chunk_map[chunk_id].storage_path).read_bytes()
                for chunk_id in tensor.chunk_ids
            )
            if len(data) != tensor.byte_length or sha256_hex(data) != tensor.checksum:
                raise IntegrityError(f"candidate tensor is corrupt: {tensor.tensor_id}")
        candidate.validate_checksum()
        return time.perf_counter() - started

    def commit(self) -> CommitResult:
        if self._state is not TransactionState.PREPARED:
            raise TransactionStateError("transaction is not prepared")
        candidate, new_chunk_data, chunks_reused, logical_bytes = self._build_candidate()
        self._check_budgets(candidate, new_chunk_data)
        intended = tuple(sorted(new_chunk_data))
        self._state = TransactionState.WRITING
        _append_journal_entry(
            self.store.root,
            transaction_id=self.transaction_id,
            state=self._state,
            parent_manifest_id=self.parent.manifest_id,
            candidate_manifest_id=candidate.manifest_id,
            intended_chunk_ids=intended,
            written_chunk_ids=(),
        )
        write_seconds = 0.0
        fsync_seconds = 0.0
        manifest_seconds = 0.0
        checksum_seconds = 0.0
        metadata_bytes_written = 0
        try:
            for chunk_id, data in sorted(new_chunk_data.items()):
                write_delta, fsync_delta = self._write_chunk(chunk_id, data)
                write_seconds += write_delta
                fsync_seconds += fsync_delta
                _append_journal_entry(
                    self.store.root,
                    transaction_id=self.transaction_id,
                    state=self._state,
                    parent_manifest_id=self.parent.manifest_id,
                    candidate_manifest_id=candidate.manifest_id,
                    intended_chunk_ids=intended,
                    written_chunk_ids=tuple(self._written_chunk_ids),
                    message=f"wrote chunk {chunk_id}",
                )
            checksum_seconds = self._validate_candidate(candidate)
            self._state = TransactionState.VALIDATED
            _append_journal_entry(
                self.store.root,
                transaction_id=self.transaction_id,
                state=self._state,
                parent_manifest_id=self.parent.manifest_id,
                candidate_manifest_id=candidate.manifest_id,
                intended_chunk_ids=intended,
                written_chunk_ids=tuple(self._written_chunk_ids),
            )
            manifest_data = canonical_json_bytes(candidate.to_dict()) + b"\n"
            manifest_path = _manifest_path(self.store.root, candidate.manifest_id)
            manifest_temporary = manifest_path.with_name(
                f".{manifest_path.name}.{self.transaction_id}.tmp"
            )
            fsync_seconds += _write_file_fsynced(manifest_temporary, manifest_data)
            self._inject(
                FailurePoint.BEFORE_MANIFEST_RENAME,
                manifest_id=candidate.manifest_id,
            )
            publish_started = time.perf_counter()
            os.replace(manifest_temporary, manifest_path)
            fsync_seconds += _fsync_directory(manifest_path.parent)
            manifest_seconds += time.perf_counter() - publish_started
            pointer_data = _current_pointer_bytes(candidate)
            pointer_temporary = self.store.root / f".CURRENT.{self.transaction_id}.tmp"
            fsync_seconds += _write_file_fsynced(pointer_temporary, pointer_data)
            self._inject(
                FailurePoint.BEFORE_CURRENT_RENAME,
                manifest_id=candidate.manifest_id,
            )
            publish_started = time.perf_counter()
            os.replace(pointer_temporary, self.store.root / "CURRENT")
            fsync_seconds += _fsync_directory(self.store.root)
            manifest_seconds += time.perf_counter() - publish_started
            metadata_bytes_written = len(manifest_data) + len(pointer_data)
            self._state = TransactionState.COMMITTED
            _append_journal_entry(
                self.store.root,
                transaction_id=self.transaction_id,
                state=self._state,
                parent_manifest_id=self.parent.manifest_id,
                candidate_manifest_id=candidate.manifest_id,
                intended_chunk_ids=intended,
                written_chunk_ids=tuple(self._written_chunk_ids),
            )
        except SimulatedCrash:
            raise
        except BaseException as exc:
            self.abort(message=f"commit failed: {type(exc).__name__}: {exc}")
            raise

        physical_chunk_bytes = sum(len(data) for data in new_chunk_data.values())
        telemetry = StoreTelemetry(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            operation="commit",
            transaction_id=self.transaction_id,
            logical_bytes_written=logical_bytes,
            physical_bytes_written=physical_chunk_bytes + metadata_bytes_written,
            chunk_writes=len(new_chunk_data),
            chunks_reused=chunks_reused,
            checksum_seconds=checksum_seconds,
            write_seconds=write_seconds,
            fsync_seconds=fsync_seconds,
            manifest_publication_seconds=manifest_seconds,
        )
        telemetry = self.store._record_telemetry(
            telemetry,
            counted_writes=physical_chunk_bytes + metadata_bytes_written,
        )
        self._payloads.clear()
        self._explicit_versions.clear()
        return CommitResult(manifest=candidate, telemetry=telemetry)


def _append_journal_entry(
    root: Path,
    *,
    transaction_id: str,
    state: TransactionState,
    parent_manifest_id: str,
    candidate_manifest_id: str | None,
    intended_chunk_ids: tuple[str, ...],
    written_chunk_ids: tuple[str, ...],
    message: str | None = None,
) -> JournalEntry:
    path = _journal_path(root, transaction_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _read_journal(path)
    entry = JournalEntry(
        schema_version=JOURNAL_SCHEMA_VERSION,
        transaction_id=transaction_id,
        sequence=len(entries),
        timestamp_utc=utc_now(),
        state=state,
        parent_manifest_id=parent_manifest_id,
        candidate_manifest_id=candidate_manifest_id,
        intended_chunk_ids=intended_chunk_ids,
        written_chunk_ids=written_chunk_ids,
        message=message,
    )
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(entry.to_dict()) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return entry


def _read_journal(path: Path) -> list[JournalEntry]:
    if not path.exists():
        return []
    entries: list[JournalEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IntegrityError(f"cannot read transaction journal: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("journal entry is not an object")
            entry = JournalEntry.from_dict(value)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IntegrityError(
                f"invalid journal entry at {path}:{line_number}"
            ) from exc
        if entry.schema_version != JOURNAL_SCHEMA_VERSION:
            raise IntegrityError(
                f"unsupported journal schema: {entry.schema_version}"
            )
        entries.append(entry)
    return entries
