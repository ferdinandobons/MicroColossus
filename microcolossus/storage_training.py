"""Observable integration between resident training and the versioned tensor store."""

from __future__ import annotations

import gc
import hashlib
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .storage import StoreLimits, TensorPayload, VersionedTensorStore
from .storage.adapters import export_pytorch_state, payload_to_torch, restore_pytorch_state
from .storage.schema import FailurePoint, StoreTelemetry, canonical_json_bytes, sha256_hex
from .telemetry import (
    accelerator_memory_metrics,
    model_checksum,
    process_rss_bytes,
    reset_accelerator_memory,
    synchronize_accelerator,
    write_json_atomic,
)
from .training import make_synthetic_lm_batch, resolve_device, seed_everything

STORAGE_STEP_SCHEMA_VERSION = "microcolossus.storage-step.v1"


@dataclass(frozen=True)
class StateDigest:
    tensor_count: int
    logical_bytes: int
    complete_checksum: str
    model_checksum: str
    optimizer_checksum: str


@dataclass(frozen=True)
class ObservedComputeStep:
    loss: float
    gradient_norm: float
    forward_seconds: float
    backward_seconds: float
    clipping_seconds: float
    optimizer_seconds: float
    total_compute_seconds: float
    process_rss_bytes: int
    accelerator_memory_measurement: str
    accelerator_allocated_bytes: int
    accelerator_driver_allocated_bytes: int
    accelerator_recommended_max_bytes: int
    parameter_checksum: str


@dataclass(frozen=True)
class StateComparison:
    names_equal: bool
    structures_equal: bool
    exact_bytes: bool
    all_values_finite: bool
    maximum_absolute_difference: float | None
    mean_absolute_difference: float | None
    maximum_relative_difference: float | None
    worst_absolute_tensor: str | None
    worst_relative_tensor: str | None
    mismatched_records: tuple[str, ...]


@dataclass(frozen=True)
class StorageReadMetrics:
    seconds: float
    logical_bytes: int
    tensor_reads: int
    referenced_chunk_reads: int


@dataclass(frozen=True)
class StorageBackedStepResult:
    schema_version: str
    experiment: str
    device: str
    store_path: str
    parameter_count: int
    batch_checksum: str
    initial_manifest_id: str
    final_manifest_id: str
    initial_state: StateDigest
    resident_final_state: StateDigest
    storage_final_state: StateDigest
    restored_final_state: StateDigest
    resident_compute: ObservedComputeStep
    storage_compute: ObservedComputeStep
    initial_store_commit: StoreTelemetry
    state_read: StorageReadMetrics
    updated_store_commit: StoreTelemetry
    storage_restore_read: StorageReadMetrics
    initial_materialization_seconds: float
    storage_materialization_seconds: float
    resident_export_seconds: float
    storage_export_seconds: float
    restored_export_seconds: float
    resident_vs_storage: StateComparison
    storage_vs_restored: StateComparison
    tensor_versions: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payload_key(payload: TensorPayload) -> str:
    return f"{payload.kind.value}:{payload.logical_name}"


def _digest_rows(payloads: tuple[TensorPayload, ...]) -> list[dict[str, Any]]:
    return [
        {
            "key": _payload_key(payload),
            "shape": list(payload.shape),
            "dtype": payload.dtype,
            "byte_order": payload.byte_order,
            "byte_length": payload.byte_length,
            "checksum": payload.checksum,
            "metadata": [list(item) for item in payload.metadata],
        }
        for payload in sorted(payloads, key=_payload_key)
    ]


def _subset_checksum(rows: list[dict[str, Any]], prefix: str) -> str:
    selected = [
        row
        for row in rows
        if str(row["key"]).split(":", 1)[1].startswith(prefix)
    ]
    return sha256_hex(canonical_json_bytes(selected))


def state_digest(payloads: tuple[TensorPayload, ...]) -> StateDigest:
    rows = _digest_rows(payloads)
    return StateDigest(
        tensor_count=len(payloads),
        logical_bytes=sum(payload.byte_length for payload in payloads),
        complete_checksum=sha256_hex(canonical_json_bytes(rows)),
        model_checksum=_subset_checksum(rows, "model"),
        optimizer_checksum=_subset_checksum(rows, "optimizer."),
    )


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    byte_view = value.reshape(-1).view(torch.uint8)
    storage = bytes(cast(Iterable[int], byte_view.untyped_storage()))
    start = byte_view.storage_offset()
    return storage[start : start + byte_view.numel()]


def batch_checksum(input_ids: Tensor, targets: Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in (("input_ids", input_ids), ("targets", targets)):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _global_gradient_norm(model: nn.Module) -> float:
    squared_norms: list[float] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_sum = parameter.grad.detach().float().pow(2).sum()
            squared_norms.append(float(squared_sum.cpu().item()))
    return math.sqrt(math.fsum(squared_norms))


def _new_pytorch_state(
    config: ExperimentConfig, device: torch.device
) -> tuple[DecoderOnlyTransformer, torch.optim.AdamW]:
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    return model, optimizer


def _release_accelerator(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    synchronize_accelerator(device)


def run_observed_pytorch_step(
    *,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.AdamW,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> ObservedComputeStep:
    model.train()
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    reset_accelerator_memory(device)
    synchronize_accelerator(device)
    total_started = time.perf_counter()

    forward_started = time.perf_counter()
    output = model(input_ids, targets)
    synchronize_accelerator(device)
    forward_seconds = time.perf_counter() - forward_started
    if output.loss is None:
        raise RuntimeError("the model did not return a training loss")

    backward_started = time.perf_counter()
    output.loss.backward()
    synchronize_accelerator(device)
    backward_seconds = time.perf_counter() - backward_started
    gradient_norm = _global_gradient_norm(model)

    clipping_started = time.perf_counter()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    synchronize_accelerator(device)
    clipping_seconds = time.perf_counter() - clipping_started

    optimizer_started = time.perf_counter()
    optimizer.step()
    synchronize_accelerator(device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    total_compute_seconds = time.perf_counter() - total_started
    memory = accelerator_memory_metrics(device)

    return ObservedComputeStep(
        loss=float(output.loss.detach().cpu().item()),
        gradient_norm=gradient_norm,
        forward_seconds=forward_seconds,
        backward_seconds=backward_seconds,
        clipping_seconds=clipping_seconds,
        optimizer_seconds=optimizer_seconds,
        total_compute_seconds=total_compute_seconds,
        process_rss_bytes=process_rss_bytes(),
        accelerator_memory_measurement=memory.measurement_kind,
        accelerator_allocated_bytes=memory.allocated_bytes,
        accelerator_driver_allocated_bytes=memory.driver_allocated_bytes,
        accelerator_recommended_max_bytes=memory.recommended_max_bytes,
        parameter_checksum=model_checksum(model),
    )


def _is_inexact_dtype(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point or dtype.is_complex


def compare_states(
    left_payloads: tuple[TensorPayload, ...],
    right_payloads: tuple[TensorPayload, ...],
) -> StateComparison:
    left = {_payload_key(payload): payload for payload in left_payloads}
    right = {_payload_key(payload): payload for payload in right_payloads}
    names_equal = set(left) == set(right)
    mismatches: list[str] = []
    exact_bytes = names_equal
    structures_equal = names_equal
    all_finite = True
    maximum_absolute = -1.0
    maximum_relative = -1.0
    worst_absolute: str | None = None
    worst_relative: str | None = None
    absolute_sum = 0.0
    compared_values = 0

    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            mismatches.append(f"missing:{name}")
            exact_bytes = False
            structures_equal = False
            continue
        left_payload = left[name]
        right_payload = right[name]
        if (
            left_payload.shape != right_payload.shape
            or left_payload.dtype != right_payload.dtype
            or left_payload.byte_order != right_payload.byte_order
            or left_payload.kind != right_payload.kind
            or left_payload.metadata != right_payload.metadata
        ):
            mismatches.append(f"structure:{name}")
            exact_bytes = False
            structures_equal = False
            continue
        if left_payload.data != right_payload.data:
            exact_bytes = False
        left_tensor = payload_to_torch(left_payload)
        right_tensor = payload_to_torch(right_payload)
        if not _is_inexact_dtype(left_tensor.dtype):
            if not torch.equal(left_tensor, right_tensor):
                mismatches.append(f"value:{name}")
            continue
        comparison_dtype = torch.complex128 if left_tensor.is_complex() else torch.float64
        left_value = left_tensor.to(comparison_dtype)
        right_value = right_tensor.to(comparison_dtype)
        finite = bool(
            torch.isfinite(left_value).all().item()
            and torch.isfinite(right_value).all().item()
        )
        all_finite = all_finite and finite
        if left_value.numel() == 0:
            continue
        absolute = (left_value - right_value).abs()
        denominator = torch.maximum(left_value.abs(), right_value.abs()).clamp_min(1e-12)
        relative = absolute / denominator
        max_absolute = float(absolute.max().item())
        max_relative = float(relative.max().item())
        absolute_sum += float(absolute.sum().item())
        compared_values += absolute.numel()
        if max_absolute > maximum_absolute:
            maximum_absolute = max_absolute
            worst_absolute = name
        if max_relative > maximum_relative:
            maximum_relative = max_relative
            worst_relative = name

    return StateComparison(
        names_equal=names_equal,
        structures_equal=structures_equal,
        exact_bytes=exact_bytes,
        all_values_finite=all_finite,
        maximum_absolute_difference=(
            max(maximum_absolute, 0.0) if compared_values else None
        ),
        mean_absolute_difference=(
            absolute_sum / compared_values if compared_values else None
        ),
        maximum_relative_difference=(
            max(maximum_relative, 0.0) if compared_values else None
        ),
        worst_absolute_tensor=worst_absolute,
        worst_relative_tensor=worst_relative,
        mismatched_records=tuple(mismatches),
    )


def _read_canonical_state(
    store: VersionedTensorStore,
    manifest_id: str | None = None,
) -> tuple[tuple[TensorPayload, ...], StorageReadMetrics]:
    manifest = (
        store.current_manifest() if manifest_id is None else store.load_manifest(manifest_id)
    )
    started = time.perf_counter()
    payloads = tuple(
        store.read_tensor(record.tensor_id, manifest.manifest_id)
        for record in manifest.tensors
    )
    return payloads, StorageReadMetrics(
        seconds=time.perf_counter() - started,
        logical_bytes=sum(record.byte_length for record in manifest.tensors),
        tensor_reads=len(manifest.tensors),
        referenced_chunk_reads=sum(len(record.chunk_ids) for record in manifest.tensors),
    )


def _store_limits(config: ExperimentConfig) -> StoreLimits:
    mib = 1024**2
    gib = 1024**3
    return StoreLimits(
        chunk_size_bytes=mib,
        max_storage_bytes=int(config.hardware.nvme_gib * gib),
        max_staging_bytes=4 * mib,
    )


def run_observable_storage_step(
    config: ExperimentConfig,
    *,
    store_path: str | Path,
    output_path: str | Path | None = None,
    device_override: str | None = None,
    update_failure_injector: Any | None = None,
) -> StorageBackedStepResult:
    """Run one resident and one storage-backed step from identical canonical state."""

    store_destination = Path(store_path)
    if store_destination.exists():
        raise FileExistsError(f"storage-step requires a new store: {store_destination}")
    device = resolve_device(device_override or config.training.device)
    seed_everything(config.training.seed)

    initial_model, initial_optimizer = _new_pytorch_state(config, torch.device("cpu"))
    initial_payloads = export_pytorch_state(initial_model, initial_optimizer)
    initial_digest = state_digest(initial_payloads)
    parameter_count = initial_model.parameter_count
    del initial_model, initial_optimizer
    gc.collect()

    store = VersionedTensorStore.create(
        store_destination,
        limits=_store_limits(config),
    )
    initial_transaction = store.begin_transaction(committed_step=0)
    initial_transaction.put_many(initial_payloads)
    initial_commit = initial_transaction.commit()

    generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 1)
    input_ids, targets = make_synthetic_lm_batch(
        batch_size=config.training.micro_batch_size,
        sequence_length=config.training.sequence_length,
        vocab_size=config.model.vocab_size,
        generator=generator,
    )
    data_checksum = batch_checksum(input_ids, targets)

    materialization_started = time.perf_counter()
    resident_model, resident_optimizer = _new_pytorch_state(config, device)
    restore_pytorch_state(resident_model, initial_payloads, resident_optimizer)
    initial_materialization_seconds = time.perf_counter() - materialization_started
    del initial_payloads
    gc.collect()

    seed_everything(config.training.seed + 2)
    resident_compute = run_observed_pytorch_step(
        model=resident_model,
        optimizer=resident_optimizer,
        input_ids=input_ids.clone(),
        targets=targets.clone(),
        device=device,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )
    export_started = time.perf_counter()
    resident_final_payloads = export_pytorch_state(resident_model, resident_optimizer)
    resident_export_seconds = time.perf_counter() - export_started
    resident_final_digest = state_digest(resident_final_payloads)
    del resident_model, resident_optimizer
    _release_accelerator(device)

    storage_payloads, read_metrics = _read_canonical_state(
        store, initial_commit.manifest.manifest_id
    )
    storage_materialization_started = time.perf_counter()
    storage_model, storage_optimizer = _new_pytorch_state(config, device)
    restore_pytorch_state(storage_model, storage_payloads, storage_optimizer)
    storage_materialization_seconds = (
        time.perf_counter() - storage_materialization_started
    )
    del storage_payloads
    gc.collect()

    seed_everything(config.training.seed + 2)
    storage_compute = run_observed_pytorch_step(
        model=storage_model,
        optimizer=storage_optimizer,
        input_ids=input_ids.clone(),
        targets=targets.clone(),
        device=device,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )
    export_started = time.perf_counter()
    storage_final_payloads = export_pytorch_state(storage_model, storage_optimizer)
    storage_export_seconds = time.perf_counter() - export_started
    storage_final_digest = state_digest(storage_final_payloads)
    resident_vs_storage = compare_states(
        resident_final_payloads, storage_final_payloads
    )

    update_transaction = store.begin_transaction(
        committed_step=1,
        failure_injector=update_failure_injector,
    )
    update_transaction.put_many(storage_final_payloads)
    updated_commit = update_transaction.commit()
    del storage_model, storage_optimizer
    _release_accelerator(device)

    restored_payloads, restored_read = _read_canonical_state(
        store, updated_commit.manifest.manifest_id
    )
    restored_model, restored_optimizer = _new_pytorch_state(config, device)
    restore_pytorch_state(restored_model, restored_payloads, restored_optimizer)
    del restored_payloads
    export_started = time.perf_counter()
    restored_final_payloads = export_pytorch_state(restored_model, restored_optimizer)
    restored_export_seconds = time.perf_counter() - export_started
    restored_final_digest = state_digest(restored_final_payloads)
    storage_vs_restored = compare_states(
        storage_final_payloads, restored_final_payloads
    )

    tensor_versions = tuple(
        sorted((record.tensor_id, record.version) for record in updated_commit.manifest.tensors)
    )
    result = StorageBackedStepResult(
        schema_version=STORAGE_STEP_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        store_path=str(store_destination),
        parameter_count=parameter_count,
        batch_checksum=data_checksum,
        initial_manifest_id=initial_commit.manifest.manifest_id,
        final_manifest_id=updated_commit.manifest.manifest_id,
        initial_state=initial_digest,
        resident_final_state=resident_final_digest,
        storage_final_state=storage_final_digest,
        restored_final_state=restored_final_digest,
        resident_compute=resident_compute,
        storage_compute=storage_compute,
        initial_store_commit=initial_commit.telemetry,
        state_read=read_metrics,
        updated_store_commit=updated_commit.telemetry,
        storage_restore_read=restored_read,
        initial_materialization_seconds=initial_materialization_seconds,
        storage_materialization_seconds=storage_materialization_seconds,
        resident_export_seconds=resident_export_seconds,
        storage_export_seconds=storage_export_seconds,
        restored_export_seconds=restored_export_seconds,
        resident_vs_storage=resident_vs_storage,
        storage_vs_restored=storage_vs_restored,
        tensor_versions=tensor_versions,
    )
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result


def fail_at(point: FailurePoint) -> Any:
    """Return a failure injector useful for deterministic integration tests."""

    from .storage import SimulatedCrash

    def inject(observed: FailurePoint, _context: dict[str, Any]) -> None:
        if observed is point:
            raise SimulatedCrash(point.value)

    return inject
