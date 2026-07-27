"""Canonical schemas for the backend-neutral tensor store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

STORE_SCHEMA_VERSION = "microcolossus.tensor-store.v1"
MANIFEST_SCHEMA_VERSION = "microcolossus.tensor-manifest.v1"
JOURNAL_SCHEMA_VERSION = "microcolossus.tensor-journal.v1"
TELEMETRY_SCHEMA_VERSION = "microcolossus.tensor-store-telemetry.v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TensorKind(StrEnum):
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    ADAM_FIRST_MOMENT = "adam_first_moment"
    ADAM_SECOND_MOMENT = "adam_second_moment"
    MASTER_WEIGHT = "master_weight"
    METADATA = "metadata"


class TransactionState(StrEnum):
    PREPARED = "prepared"
    WRITING = "writing"
    VALIDATED = "validated"
    COMMITTED = "committed"
    ABORTED = "aborted"


class FailurePoint(StrEnum):
    BEFORE_CHUNK_WRITE = "before_chunk_write"
    DURING_CHUNK_WRITE = "during_chunk_write"
    BEFORE_CHUNK_FSYNC = "before_chunk_fsync"
    BEFORE_MANIFEST_RENAME = "before_manifest_rename"
    BEFORE_CURRENT_RENAME = "before_current_rename"


@dataclass(frozen=True)
class StoreLimits:
    """Hard logical limits enforced by the synchronous store."""

    chunk_size_bytes: int = 4 * 1024 * 1024
    max_storage_bytes: int = 100 * 1024**3
    max_staging_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for value, name in (
            (self.chunk_size_bytes, "chunk_size_bytes"),
            (self.max_storage_bytes, "max_storage_bytes"),
            (self.max_staging_bytes, "max_staging_bytes"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.chunk_size_bytes > self.max_staging_bytes:
            raise ValueError("chunk_size_bytes cannot exceed max_staging_bytes")


@dataclass(frozen=True)
class StoreMetadata:
    schema_version: str
    store_id: str
    created_at_utc: str
    limits: StoreLimits

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StoreMetadata:
        limits = StoreLimits(**dict(value["limits"]))
        return cls(
            schema_version=str(value["schema_version"]),
            store_id=str(value["store_id"]),
            created_at_utc=str(value["created_at_utc"]),
            limits=limits,
        )


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    storage_path: str
    byte_offset: int
    byte_length: int
    checksum: str
    compression: str
    creating_transaction: str

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if self.byte_offset < 0:
            raise ValueError("byte_offset cannot be negative")
        if self.byte_length < 0:
            raise ValueError("byte_length cannot be negative")
        if self.compression != "none":
            raise ValueError("only compression='none' is currently supported")
        path = PurePosixPath(self.storage_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("chunk storage_path must remain inside the store")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChunkRecord:
        return cls(
            chunk_id=str(value["chunk_id"]),
            storage_path=str(value["storage_path"]),
            byte_offset=int(value["byte_offset"]),
            byte_length=int(value["byte_length"]),
            checksum=str(value["checksum"]),
            compression=str(value["compression"]),
            creating_transaction=str(value["creating_transaction"]),
        )


@dataclass(frozen=True)
class TensorRecord:
    tensor_id: str
    logical_name: str
    kind: TensorKind
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    version: int
    chunk_ids: tuple[str, ...]
    byte_length: int
    checksum: str
    committed_step: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("tensor_id cannot be empty")
        if not self.logical_name:
            raise ValueError("logical_name cannot be empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor shape dimensions cannot be negative")
        if self.byte_order not in {"little", "not_applicable"}:
            raise ValueError("byte_order must be little or not_applicable")
        if self.version < 0:
            raise ValueError("version cannot be negative")
        if self.byte_length < 0:
            raise ValueError("byte_length cannot be negative")
        if self.committed_step < -1:
            raise ValueError("committed_step cannot be less than -1")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["shape"] = list(self.shape)
        value["chunk_ids"] = list(self.chunk_ids)
        value["metadata"] = [list(item) for item in self.metadata]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TensorRecord:
        return cls(
            tensor_id=str(value["tensor_id"]),
            logical_name=str(value["logical_name"]),
            kind=TensorKind(str(value["kind"])),
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            byte_order=str(value["byte_order"]),
            version=int(value["version"]),
            chunk_ids=tuple(str(item) for item in value["chunk_ids"]),
            byte_length=int(value["byte_length"]),
            checksum=str(value["checksum"]),
            committed_step=int(value["committed_step"]),
            metadata=tuple(
                (str(item[0]), str(item[1])) for item in value.get("metadata", [])
            ),
        )


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    manifest_id: str
    parent_manifest_id: str | None
    committed_step: int
    created_at_utc: str
    tensors: tuple[TensorRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    aggregate_logical_bytes: int
    aggregate_physical_bytes: int
    manifest_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "parent_manifest_id": self.parent_manifest_id,
            "committed_step": self.committed_step,
            "created_at_utc": self.created_at_utc,
            "tensors": [item.to_dict() for item in self.tensors],
            "chunks": [item.to_dict() for item in self.chunks],
            "aggregate_logical_bytes": self.aggregate_logical_bytes,
            "aggregate_physical_bytes": self.aggregate_physical_bytes,
        }

    def validate_structure(self) -> None:
        tensor_ids = [item.tensor_id for item in self.tensors]
        chunk_ids = [item.chunk_id for item in self.chunks]
        if len(tensor_ids) != len(set(tensor_ids)):
            raise ValueError("manifest contains duplicate tensor IDs")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("manifest contains duplicate chunk IDs")
        chunk_map = {item.chunk_id: item for item in self.chunks}
        for chunk in self.chunks:
            if chunk.chunk_id != chunk.checksum:
                raise ValueError("content-addressed chunk ID must equal its checksum")
        for tensor in self.tensors:
            missing = [item for item in tensor.chunk_ids if item not in chunk_map]
            if missing:
                raise ValueError(
                    f"tensor {tensor.tensor_id} references missing chunks: {missing}"
                )
            chunk_bytes = sum(chunk_map[item].byte_length for item in tensor.chunk_ids)
            if chunk_bytes != tensor.byte_length:
                raise ValueError(
                    f"tensor {tensor.tensor_id} chunk lengths do not match byte_length"
                )
        if self.aggregate_logical_bytes != sum(
            item.byte_length for item in self.tensors
        ):
            raise ValueError("aggregate_logical_bytes is inconsistent")
        if self.aggregate_physical_bytes != sum(
            item.byte_length for item in self.chunks
        ):
            raise ValueError("aggregate_physical_bytes is inconsistent")

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_computed_checksum(self) -> Manifest:
        return replace(self, manifest_checksum=self.compute_checksum())

    def validate_checksum(self) -> None:
        expected = self.compute_checksum()
        if self.manifest_checksum != expected:
            raise ValueError(
                f"manifest checksum mismatch: {self.manifest_checksum} != {expected}"
            )

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["manifest_checksum"] = self.manifest_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Manifest:
        return cls(
            schema_version=str(value["schema_version"]),
            manifest_id=str(value["manifest_id"]),
            parent_manifest_id=(
                str(value["parent_manifest_id"])
                if value.get("parent_manifest_id") is not None
                else None
            ),
            committed_step=int(value["committed_step"]),
            created_at_utc=str(value["created_at_utc"]),
            tensors=tuple(TensorRecord.from_dict(dict(item)) for item in value["tensors"]),
            chunks=tuple(ChunkRecord.from_dict(dict(item)) for item in value["chunks"]),
            aggregate_logical_bytes=int(value["aggregate_logical_bytes"]),
            aggregate_physical_bytes=int(value["aggregate_physical_bytes"]),
            manifest_checksum=str(value["manifest_checksum"]),
        )


@dataclass(frozen=True)
class JournalEntry:
    schema_version: str
    transaction_id: str
    sequence: int
    timestamp_utc: str
    state: TransactionState
    parent_manifest_id: str
    candidate_manifest_id: str | None
    intended_chunk_ids: tuple[str, ...]
    written_chunk_ids: tuple[str, ...]
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        value["intended_chunk_ids"] = list(self.intended_chunk_ids)
        value["written_chunk_ids"] = list(self.written_chunk_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JournalEntry:
        return cls(
            schema_version=str(value["schema_version"]),
            transaction_id=str(value["transaction_id"]),
            sequence=int(value["sequence"]),
            timestamp_utc=str(value["timestamp_utc"]),
            state=TransactionState(str(value["state"])),
            parent_manifest_id=str(value["parent_manifest_id"]),
            candidate_manifest_id=(
                str(value["candidate_manifest_id"])
                if value.get("candidate_manifest_id") is not None
                else None
            ),
            intended_chunk_ids=tuple(str(item) for item in value["intended_chunk_ids"]),
            written_chunk_ids=tuple(str(item) for item in value["written_chunk_ids"]),
            message=(str(value["message"]) if value.get("message") is not None else None),
        )


@dataclass(frozen=True)
class StoreTelemetry:
    schema_version: str
    operation: str
    transaction_id: str | None
    logical_bytes_read: int = 0
    logical_bytes_written: int = 0
    physical_bytes_read: int = 0
    physical_bytes_written: int = 0
    chunk_reads: int = 0
    chunk_writes: int = 0
    chunks_reused: int = 0
    checksum_seconds: float = 0.0
    read_seconds: float = 0.0
    write_seconds: float = 0.0
    fsync_seconds: float = 0.0
    manifest_publication_seconds: float = 0.0
    recovery_seconds: float = 0.0
    store_size_bytes: int = 0
    cumulative_physical_bytes_written: int = 0
    actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["actions"] = list(self.actions)
        return value
