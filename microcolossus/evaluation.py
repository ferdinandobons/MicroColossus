"""Validation loss, deterministic samples, and atomic progress records."""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .bounded_backward import _batch_checksum
from .config import ExperimentConfig
from .data import PreparedDataSource
from .model import DecoderOnlyTransformer
from .step_bundle import StepBundleManifest, StepBundleStore
from .storage.adapters import restore_pytorch_model
from .storage.schema import canonical_json_bytes
from .telemetry import (
    AcceleratorMemoryMetrics,
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
)
from .training_checkpoint import _open_referenced_store

EVALUATION_SCHEMA_VERSION = "microcolossus.evaluation.v1"
PROGRESS_SCHEMA_VERSION = "microcolossus.training-progress.v1"


@dataclass(frozen=True)
class EvaluationResult:
    schema_version: str
    step: int
    bundle_id: str
    data_identity_checksum: str
    validation_loss: float
    validation_batch_checksums: tuple[str, ...]
    validation_token_count: int
    sample_prompt: str
    sample_completion: str
    sample_token_ids: tuple[int, ...]
    evaluation_seconds: float
    process_rss_bytes: int
    accelerator_memory: AcceleratorMemoryMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationResult:
        memory = value["accelerator_memory"]
        return cls(
            schema_version=str(value["schema_version"]),
            step=int(value["step"]),
            bundle_id=str(value["bundle_id"]),
            data_identity_checksum=str(value["data_identity_checksum"]),
            validation_loss=float(value["validation_loss"]),
            validation_batch_checksums=tuple(
                str(item) for item in value["validation_batch_checksums"]
            ),
            validation_token_count=int(value["validation_token_count"]),
            sample_prompt=str(value["sample_prompt"]),
            sample_completion=str(value["sample_completion"]),
            sample_token_ids=tuple(int(item) for item in value["sample_token_ids"]),
            evaluation_seconds=float(value["evaluation_seconds"]),
            process_rss_bytes=int(value["process_rss_bytes"]),
            accelerator_memory=AcceleratorMemoryMetrics(
                measurement_kind=str(memory["measurement_kind"]),
                allocated_bytes=int(memory["allocated_bytes"]),
                driver_allocated_bytes=int(memory["driver_allocated_bytes"]),
                recommended_max_bytes=int(memory["recommended_max_bytes"]),
            ),
        )


@dataclass(frozen=True)
class TrainingProgressRecord:
    schema_version: str
    step: int
    bundle_id: str
    parent_bundle_id: str | None
    batch_cursor: int | None
    batch_seed: int | None
    batch_source_kind: str
    batch_offsets: tuple[int, ...]
    batch_checksum: str
    training_loss: float | None
    gradient_norm: float | None
    clipping_coefficient: float | None
    evaluation: EvaluationResult | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingProgressRecord:
        evaluation_value = value.get("evaluation")
        return cls(
            schema_version=str(value["schema_version"]),
            step=int(value["step"]),
            bundle_id=str(value["bundle_id"]),
            parent_bundle_id=(
                None if value.get("parent_bundle_id") is None else str(value["parent_bundle_id"])
            ),
            batch_cursor=(
                None if value.get("batch_cursor") is None else int(value["batch_cursor"])
            ),
            batch_seed=(
                None if value.get("batch_seed") is None else int(value["batch_seed"])
            ),
            batch_source_kind=str(value["batch_source_kind"]),
            batch_offsets=tuple(int(item) for item in value["batch_offsets"]),
            batch_checksum=str(value["batch_checksum"]),
            training_loss=(
                None if value.get("training_loss") is None else float(value["training_loss"])
            ),
            gradient_norm=(
                None if value.get("gradient_norm") is None else float(value["gradient_norm"])
            ),
            clipping_coefficient=(
                None
                if value.get("clipping_coefficient") is None
                else float(value["clipping_coefficient"])
            ),
            evaluation=(
                None
                if evaluation_value is None
                else EvaluationResult.from_dict(evaluation_value)
            ),
        )


def _load_model(
    config: ExperimentConfig,
    *,
    bundle_store: StepBundleStore,
    manifest: StepBundleManifest,
    device: torch.device,
) -> DecoderOnlyTransformer:
    parameter_store = _open_referenced_store(bundle_store, manifest, kind="parameter")
    payloads = tuple(
        parameter_store.read_tensor(record.tensor_id)
        for record in sorted(
            parameter_store.current_manifest().tensors,
            key=lambda item: item.logical_name,
        )
    )
    model = DecoderOnlyTransformer(config.model).to(device)
    restore_pytorch_model(model, payloads)
    del payloads
    model.eval()
    return model


def _generate(
    model: DecoderOnlyTransformer,
    *,
    data_source: PreparedDataSource,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, tuple[int, ...]]:
    if max_new_tokens <= 0:
        return "", ()
    prompt_ids = list(data_source.encode_text(prompt))
    if not prompt_ids:
        prompt_ids = list(data_source.encode_text(" "))
    context_limit = model.config.max_sequence_length
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = (prompt_ids + generated)[-context_limit:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids).logits
            token = int(torch.argmax(logits[0, -1]).detach().cpu().item())
            generated.append(token)
    synchronize_accelerator(device)
    return data_source.decode_tokens(generated), tuple(generated)


def evaluate_committed_bundle(
    config: ExperimentConfig,
    *,
    bundle_store: StepBundleStore,
    manifest: StepBundleManifest,
    data_source: PreparedDataSource,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate validation loss and an optional deterministic greedy sample."""

    started = time.perf_counter()
    model = _load_model(
        config,
        bundle_store=bundle_store,
        manifest=manifest,
        device=device,
    )
    loss_sum = 0.0
    token_count = 0
    checksums: list[str] = []
    with torch.no_grad():
        for cursor in range(config.evaluation.validation_batches):
            batch = data_source.validation_batch(cursor)
            checksums.append(_batch_checksum(batch.input_ids, batch.targets))
            output = model(batch.input_ids.to(device), batch.targets.to(device))
            if output.loss is None:
                raise RuntimeError("validation model did not return a loss")
            tokens = batch.targets.numel()
            loss_sum += float(output.loss.detach().cpu().item()) * tokens
            token_count += tokens
    synchronize_accelerator(device)
    prompt = config.evaluation.sample_prompt
    if not prompt and config.evaluation.sample_tokens > 0:
        prompt = data_source.default_prompt(
            min(32, max(1, config.model.max_sequence_length // 2))
        )
    completion, token_ids = _generate(
        model,
        data_source=data_source,
        prompt=prompt,
        max_new_tokens=config.evaluation.sample_tokens,
        device=device,
    )
    result = EvaluationResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        step=manifest.committed_step,
        bundle_id=manifest.bundle_id,
        data_identity_checksum=data_source.identity.identity_checksum,
        validation_loss=loss_sum / token_count,
        validation_batch_checksums=tuple(checksums),
        validation_token_count=token_count,
        sample_prompt=prompt,
        sample_completion=completion,
        sample_token_ids=token_ids,
        evaluation_seconds=time.perf_counter() - started,
        process_rss_bytes=process_rss_bytes(),
        accelerator_memory=accelerator_memory_metrics(device),
    )
    del model
    gc.collect()
    synchronize_accelerator(device)
    return result


def progress_path(root: Path, step: int) -> Path:
    return root / "metrics" / f"step-{step:08d}.json"


def write_progress_record(root: Path, record: TrainingProgressRecord) -> Path:
    path = progress_path(root, record.step)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = canonical_json_bytes(record.to_dict()) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def load_progress_record(path: Path) -> TrainingProgressRecord:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid progress record: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("progress record must be a JSON object")
    record = TrainingProgressRecord.from_dict(value)
    if record.schema_version != PROGRESS_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported progress schema: {record.schema_version}")
    return record


def load_progress_records(root: Path) -> tuple[TrainingProgressRecord, ...]:
    metrics = root / "metrics"
    if not metrics.exists():
        return ()
    records = tuple(load_progress_record(path) for path in sorted(metrics.glob("step-*.json")))
    if tuple(record.step for record in records) != tuple(range(len(records))):
        raise RuntimeError("progress records are not contiguous from step zero")
    return records


def should_evaluate(config: ExperimentConfig, step: int) -> bool:
    return config.evaluation.enabled and step % config.evaluation.interval_steps == 0


def ensure_progress_record(
    config: ExperimentConfig,
    *,
    bundle_store: StepBundleStore,
    manifest: StepBundleManifest,
    data_source: PreparedDataSource,
    device: torch.device,
    batch_cursor: int | None,
    batch_seed: int | None,
    batch_source_kind: str,
    batch_offsets: tuple[int, ...],
    batch_checksum: str,
    training_loss: float | None,
    gradient_norm: float | None,
    clipping_coefficient: float | None,
) -> TrainingProgressRecord:
    """Create or validate the derived record for one committed root bundle."""

    path = progress_path(bundle_store.root, manifest.committed_step)
    if path.exists():
        record = load_progress_record(path)
        if record.bundle_id != manifest.bundle_id:
            raise RuntimeError("progress record bundle does not match authoritative bundle")
        return record
    evaluation = (
        evaluate_committed_bundle(
            config,
            bundle_store=bundle_store,
            manifest=manifest,
            data_source=data_source,
            device=device,
        )
        if should_evaluate(config, manifest.committed_step)
        else None
    )
    record = TrainingProgressRecord(
        schema_version=PROGRESS_SCHEMA_VERSION,
        step=manifest.committed_step,
        bundle_id=manifest.bundle_id,
        parent_bundle_id=manifest.parent_bundle_id,
        batch_cursor=batch_cursor,
        batch_seed=batch_seed,
        batch_source_kind=batch_source_kind,
        batch_offsets=batch_offsets,
        batch_checksum=batch_checksum,
        training_loss=training_loss,
        gradient_norm=gradient_norm,
        clipping_coefficient=clipping_coefficient,
        evaluation=evaluation,
    )
    write_progress_record(bundle_store.root, record)
    return record
