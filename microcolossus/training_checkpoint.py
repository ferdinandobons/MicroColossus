"""Persistent training metadata, data cursor, and bundle initialization."""

from __future__ import annotations

import gc
import json
import os
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from torch import Tensor

from .bounded_optimizer import _initialize_adamw_payloads, _store_limits
from .config import ExperimentConfig
from .data import DataIdentity, PreparedDataSource, prepare_data_source
from .model import DecoderOnlyTransformer
from .step_bundle import (
    BundlePublicationTelemetry,
    StepBundleManifest,
    StepBundleStore,
)
from .storage import IntegrityError, TensorPayload, VersionedTensorStore
from .storage.adapters import export_pytorch_model
from .storage.schema import StoreTelemetry, canonical_json_bytes, sha256_hex
from .training import seed_everything

BOUNDED_TRAINING_SCHEMA_VERSION = "microcolossus.bounded-training.v3"
TRAINING_METADATA_SCHEMA_VERSION = "microcolossus.training-metadata.v2"
MULTI_STEP_RUNTIME_VERSION = "0.10.0"
ACTIVATION_RECOMPUTE_RUNTIME_VERSION = "0.12.0"
HYBRID_ACTIVATION_RUNTIME_VERSION = "0.13.0"
BATCH_STREAM_VERSION = "configured-data-source-v1"
SCHEDULE_KIND = "constant"


class ResumeConfigurationError(RuntimeError):
    """Raised when an existing training root is incompatible with the requested run."""


@dataclass(frozen=True)
class TrainingMetadata:
    schema_version: str
    config_digest: str
    runtime_version: str
    seed: int
    batch_stream: str
    schedule_kind: str
    data_identity: DataIdentity
    metadata_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_digest": self.config_digest,
            "runtime_version": self.runtime_version,
            "seed": self.seed,
            "batch_stream": self.batch_stream,
            "schedule_kind": self.schedule_kind,
            "data_identity": self.data_identity.to_dict(),
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_checksum(self) -> TrainingMetadata:
        return TrainingMetadata(
            schema_version=self.schema_version,
            config_digest=self.config_digest,
            runtime_version=self.runtime_version,
            seed=self.seed,
            batch_stream=self.batch_stream,
            schedule_kind=self.schedule_kind,
            data_identity=self.data_identity,
            metadata_checksum=self.compute_checksum(),
        )

    def validate(self) -> None:
        if self.schema_version != TRAINING_METADATA_SCHEMA_VERSION:
            raise IntegrityError(
                f"unsupported training metadata schema: {self.schema_version}"
            )
        self.data_identity.validate()
        if self.metadata_checksum != self.compute_checksum():
            raise IntegrityError("training metadata checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["metadata_checksum"] = self.metadata_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingMetadata:
        result = cls(
            schema_version=str(value["schema_version"]),
            config_digest=str(value["config_digest"]),
            runtime_version=str(value["runtime_version"]),
            seed=int(value["seed"]),
            batch_stream=str(value["batch_stream"]),
            schedule_kind=str(value["schedule_kind"]),
            data_identity=DataIdentity.from_dict(dict(value["data_identity"])),
            metadata_checksum=str(value["metadata_checksum"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class BundleLineageEntry:
    bundle_id: str
    parent_bundle_id: str | None
    committed_step: int
    batch_checksum: str
    parameter_manifest_id: str
    optimizer_manifest_id: str
    gradient_manifest_id: str | None


def _semantic_config(config: ExperimentConfig) -> dict[str, Any]:
    training: dict[str, Any] = {
        "micro_batch_size": config.training.micro_batch_size,
        "sequence_length": config.training.sequence_length,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "gradient_clip_norm": config.training.gradient_clip_norm,
        "seed": config.training.seed,
        "mode": config.training.mode,
    }
    if config.training.activation_policy != "retain_all":
        training["activation_policy"] = config.training.activation_policy
    if config.training.activation_policy == "hybrid":
        training["activation_anchor_policy"] = asdict(
            config.training.activation_anchor_policy
        )
    return {
        "model": asdict(config.model),
        "training": training,
        "data": {
            "kind": config.data.kind,
            "validation_fraction": config.data.validation_fraction,
            "tokenizer": config.data.tokenizer,
            "sampler": config.data.sampler,
            "separate_validation_file": config.data.validation_path is not None,
        },
        "evaluation": asdict(config.evaluation),
        "batch_stream": BATCH_STREAM_VERSION,
        "schedule_kind": SCHEDULE_KIND,
    }


def config_digest(config: ExperimentConfig) -> str:
    return sha256_hex(canonical_json_bytes(_semantic_config(config)))


def _metadata_for(
    config: ExperimentConfig,
    data_source: PreparedDataSource,
) -> TrainingMetadata:
    if config.training.activation_policy == "recompute":
        runtime_version = ACTIVATION_RECOMPUTE_RUNTIME_VERSION
    elif config.training.activation_policy == "hybrid":
        runtime_version = HYBRID_ACTIVATION_RUNTIME_VERSION
    else:
        runtime_version = MULTI_STEP_RUNTIME_VERSION
    return TrainingMetadata(
        schema_version=TRAINING_METADATA_SCHEMA_VERSION,
        config_digest=config_digest(config),
        runtime_version=runtime_version,
        seed=config.training.seed,
        batch_stream=data_source.identity.batch_stream_version,
        schedule_kind=SCHEDULE_KIND,
        data_identity=data_source.identity,
    ).with_checksum()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_training_metadata(root: Path, metadata: TrainingMetadata) -> None:
    path = root / "TRAINING.json"
    temporary = root / f".TRAINING.{uuid.uuid4().hex}.tmp"
    data = canonical_json_bytes(metadata.to_dict()) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(root)


def _load_training_metadata(root: Path) -> TrainingMetadata:
    path = root / "TRAINING.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("training metadata must be a JSON object")
        return TrainingMetadata.from_dict(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"cannot read valid training metadata from {path}") from exc


def _validate_resume_metadata(
    config: ExperimentConfig,
    metadata: TrainingMetadata,
    data_source: PreparedDataSource,
) -> None:
    expected = _metadata_for(config, data_source)
    mismatches: list[str] = []
    for name in (
        "config_digest",
        "runtime_version",
        "seed",
        "batch_stream",
        "schedule_kind",
    ):
        if getattr(metadata, name) != getattr(expected, name):
            mismatches.append(name)
    if metadata.data_identity.to_dict() != expected.data_identity.to_dict():
        mismatches.append("data_identity")
    if mismatches:
        raise ResumeConfigurationError(
            "training metadata does not match the requested configuration: "
            + ", ".join(mismatches)
        )


@lru_cache(maxsize=8)
def _cached_data_source(config: ExperimentConfig) -> PreparedDataSource:
    return prepare_data_source(config)


def _prepare_data_source_for_run(config: ExperimentConfig) -> PreparedDataSource:
    """Start one invocation with a fresh immutable view of configured data."""

    _cached_data_source.cache_clear()
    return _cached_data_source(config)


def _batch_for_cursor(config: ExperimentConfig, cursor: int) -> tuple[Tensor, Tensor, int]:
    """Compatibility wrapper for callers that need one configured training batch."""

    batch = _cached_data_source(config).training_batch(cursor)
    return batch.input_ids, batch.targets, batch.seed


def _open_referenced_store(
    bundle_store: StepBundleStore,
    manifest: StepBundleManifest,
    *,
    kind: str,
) -> VersionedTensorStore:
    reference = manifest.parameter_store if kind == "parameter" else manifest.optimizer_store
    store = VersionedTensorStore.open(bundle_store.root / reference.path)
    current = store.current_manifest()
    if current.manifest_id != reference.manifest_id:
        raise IntegrityError(f"{kind} store current manifest differs from bundle reference")
    if current.manifest_checksum != reference.manifest_checksum:
        raise IntegrityError(f"{kind} store checksum differs from bundle reference")
    store.verify(reference.manifest_id)
    return store


def _lineage(bundle_store: StepBundleStore) -> tuple[BundleLineageEntry, ...]:
    current = bundle_store.current_manifest()
    manifests: list[StepBundleManifest] = []
    seen: set[str] = set()
    candidate: StepBundleManifest | None = current
    while candidate is not None:
        if candidate.bundle_id in seen:
            raise IntegrityError("cycle detected in step-bundle lineage")
        seen.add(candidate.bundle_id)
        manifests.append(candidate)
        candidate = (
            None
            if candidate.parent_bundle_id is None
            else bundle_store.load_manifest(candidate.parent_bundle_id)
        )
    manifests.reverse()
    for expected_step, manifest in enumerate(manifests):
        if manifest.committed_step != expected_step:
            raise IntegrityError("step-bundle lineage contains a non-contiguous step")
        if expected_step == 0 and manifest.parent_bundle_id is not None:
            raise IntegrityError("step-zero bundle cannot have a parent")
        if (
            expected_step > 0
            and manifest.parent_bundle_id != manifests[expected_step - 1].bundle_id
        ):
            raise IntegrityError("step-bundle parent lineage is inconsistent")
    return tuple(
        BundleLineageEntry(
            bundle_id=item.bundle_id,
            parent_bundle_id=item.parent_bundle_id,
            committed_step=item.committed_step,
            batch_checksum=item.batch_checksum,
            parameter_manifest_id=item.parameter_store.manifest_id,
            optimizer_manifest_id=item.optimizer_store.manifest_id,
            gradient_manifest_id=(
                None if item.gradient_store is None else item.gradient_store.manifest_id
            ),
        )
        for item in manifests
    )


def _commit_initial_payloads(
    store: VersionedTensorStore,
    payloads: tuple[TensorPayload, ...],
) -> StoreTelemetry:
    transaction = store.begin_transaction(committed_step=0)
    transaction.put_many(payloads)
    return transaction.commit().telemetry


def _initialize_training_root(
    config: ExperimentConfig,
    destination: Path,
    data_source: PreparedDataSource,
) -> tuple[StepBundleStore, StepBundleManifest, BundlePublicationTelemetry, TrainingMetadata]:
    seed_everything(config.training.seed)
    bundle_store = StepBundleStore.create(destination)
    metadata = _metadata_for(config, data_source)
    _write_training_metadata(destination, metadata)

    parameter_store_path = destination / "candidates" / "step-0-parameters"
    optimizer_store_path = destination / "candidates" / "step-0-optimizer"
    parameter_store = VersionedTensorStore.create(
        parameter_store_path,
        limits=_store_limits(config),
    )
    optimizer_store = VersionedTensorStore.create(
        optimizer_store_path,
        limits=_store_limits(config),
    )

    model = DecoderOnlyTransformer(config.model)
    parameter_payloads = export_pytorch_model(model)
    del model
    optimizer_payloads = _initialize_adamw_payloads(config)
    _commit_initial_payloads(parameter_store, parameter_payloads)
    _commit_initial_payloads(optimizer_store, optimizer_payloads)
    del parameter_payloads, optimizer_payloads
    gc.collect()

    initial, publication = bundle_store.publish(
        committed_step=0,
        parameter_store_path=parameter_store_path,
        optimizer_store_path=optimizer_store_path,
        gradient_store_path=None,
        batch_checksum="",
    )
    bundle_store.verify(initial.bundle_id)
    return bundle_store, initial, publication, metadata


def _parameter_payloads(store: VersionedTensorStore) -> tuple[TensorPayload, ...]:
    return tuple(
        store.read_tensor(record.tensor_id)
        for record in sorted(store.current_manifest().tensors, key=lambda item: item.logical_name)
    )
