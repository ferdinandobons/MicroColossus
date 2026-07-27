"""Bounded parameter-group forward execution backed by the tensor store."""

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
from torch import Tensor
from torch.nn import functional as F

from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .storage import StoreLimits, TensorPayload, VersionedTensorStore
from .storage.adapters import export_pytorch_model, payload_to_torch, restore_pytorch_state
from .storage.schema import TensorRecord
from .telemetry import (
    AcceleratorMemoryMetrics,
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training import make_synthetic_lm_batch, resolve_device, seed_everything

BOUNDED_FORWARD_SCHEMA_VERSION = "microcolossus.bounded-forward.v1"


class WorkingSetExceededError(RuntimeError):
    """Raised when one execution group exceeds the declared parameter budget."""


@dataclass(frozen=True)
class TensorComparison:
    exact_bytes: bool
    all_values_finite: bool
    maximum_absolute_difference: float
    mean_absolute_difference: float
    maximum_relative_difference: float


@dataclass(frozen=True)
class ExecutionGroupSpec:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionGroupMetrics:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    tensor_count: int
    referenced_chunk_reads: int
    logical_parameter_bytes: int
    read_seconds: float
    materialization_seconds: float
    compute_seconds: float
    release_seconds: float
    input_activation_bytes: int
    output_activation_bytes: int
    output_checksum: str
    resident_comparison: TensorComparison
    process_rss_after_compute_bytes: int
    accelerator_after_materialization: AcceleratorMemoryMetrics
    accelerator_after_compute: AcceleratorMemoryMetrics
    accelerator_after_release: AcceleratorMemoryMetrics


@dataclass(frozen=True)
class BoundedForwardResult:
    schema_version: str
    experiment: str
    device: str
    store_path: str
    manifest_id: str
    manifest_checksum: str
    parameter_count: int
    batch_checksum: str
    parameter_working_set_budget_bytes: int
    maximum_group_parameter_bytes: int
    budget_respected: bool
    bootstrap_model_released_before_bounded: bool
    resident_model_released_before_bounded: bool
    retained_activations_during_bounded: bool
    total_tensor_reads: int
    unique_tensor_reads: int
    repeated_tensor_reads: int
    total_referenced_chunk_reads: int
    total_logical_parameter_bytes_read: int
    resident_loss: float
    bounded_loss: float
    loss_absolute_difference: float
    resident_logits_checksum: str
    bounded_logits_checksum: str
    logits_comparison: TensorComparison
    execution_groups: tuple[ExecutionGroupMetrics, ...]
    manifest_unchanged: bool
    store_verified_tensor_count: int
    store_verified_chunk_count: int
    store_verified_logical_bytes: int
    store_verified_physical_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ResidentTrace:
    loss: float
    logits: Tensor
    boundaries: dict[str, Tensor]


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    byte_view = value.reshape(-1).view(torch.uint8)
    storage = bytes(cast(Iterable[int], byte_view.untyped_storage()))
    start = byte_view.storage_offset()
    return storage[start : start + byte_view.numel()]


def tensor_checksum(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def batch_checksum(input_ids: Tensor, targets: Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in (("input_ids", input_ids), ("targets", targets)):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(_tensor_bytes(contiguous))
    return digest.hexdigest()


def _activation_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def compare_tensors(left: Tensor, right: Tensor) -> TensorComparison:
    left_cpu = left.detach().cpu().contiguous()
    right_cpu = right.detach().cpu().contiguous()
    if left_cpu.shape != right_cpu.shape:
        raise ValueError(
            f"tensor shape mismatch: {tuple(left_cpu.shape)} != {tuple(right_cpu.shape)}"
        )
    exact_bytes = (
        left_cpu.dtype == right_cpu.dtype and _tensor_bytes(left_cpu) == _tensor_bytes(right_cpu)
    )
    if left_cpu.numel() == 0:
        return TensorComparison(
            exact_bytes=exact_bytes,
            all_values_finite=True,
            maximum_absolute_difference=0.0,
            mean_absolute_difference=0.0,
            maximum_relative_difference=0.0,
        )
    comparison_dtype = torch.complex128 if left_cpu.is_complex() else torch.float64
    left_value = left_cpu.to(comparison_dtype)
    right_value = right_cpu.to(comparison_dtype)
    all_finite = bool(
        torch.isfinite(left_value).all().item()
        and torch.isfinite(right_value).all().item()
    )
    absolute = (left_value - right_value).abs()
    denominator = torch.maximum(left_value.abs(), right_value.abs()).clamp_min(1e-12)
    relative = absolute / denominator
    return TensorComparison(
        exact_bytes=exact_bytes,
        all_values_finite=all_finite,
        maximum_absolute_difference=float(absolute.max().item()),
        mean_absolute_difference=float(absolute.mean().item()),
        maximum_relative_difference=float(relative.max().item()),
    )


def _store_limits(config: ExperimentConfig) -> StoreLimits:
    mib = 1024**2
    gib = 1024**3
    return StoreLimits(
        chunk_size_bytes=mib,
        max_storage_bytes=int(config.hardware.nvme_gib * gib),
        max_staging_bytes=4 * mib,
    )


def _release_accelerator(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    synchronize_accelerator(device)


def _record_map(records: tuple[TensorRecord, ...]) -> dict[str, TensorRecord]:
    by_name: dict[str, TensorRecord] = {}
    for record in records:
        if record.logical_name in by_name:
            raise ValueError(f"duplicate logical tensor name: {record.logical_name}")
        by_name[record.logical_name] = record
    return by_name


def build_execution_groups(
    config: ExperimentConfig,
    available_names: set[str],
) -> tuple[ExecutionGroupSpec, ...]:
    groups: list[ExecutionGroupSpec] = [
        ExecutionGroupSpec(
            ordinal=0,
            name="embedding",
            tensor_names=(
                "model.token_embedding.weight",
                "model.position_embedding.weight",
            ),
        )
    ]
    for index in range(config.model.layers):
        prefix = f"model.blocks.{index}"
        groups.append(
            ExecutionGroupSpec(
                ordinal=len(groups),
                name=f"block-{index}",
                tensor_names=(
                    f"{prefix}.attention_norm.weight",
                    f"{prefix}.attention_norm.bias",
                    f"{prefix}.attention.qkv.weight",
                    f"{prefix}.attention.output.weight",
                    f"{prefix}.mlp_norm.weight",
                    f"{prefix}.mlp_norm.bias",
                    f"{prefix}.mlp.input.weight",
                    f"{prefix}.mlp.output.weight",
                ),
            )
        )
    head_name = (
        "model.lm_head.weight"
        if "model.lm_head.weight" in available_names
        else "model.token_embedding.weight"
    )
    groups.append(
        ExecutionGroupSpec(
            ordinal=len(groups),
            name="final-head",
            tensor_names=(
                "model.final_norm.weight",
                "model.final_norm.bias",
                head_name,
            ),
        )
    )
    required = {name for group in groups for name in group.tensor_names}
    missing = sorted(required - available_names)
    if missing:
        raise KeyError(f"missing tensors for bounded forward: {missing}")
    return tuple(groups)


def _resident_trace(
    config: ExperimentConfig,
    payloads: tuple[TensorPayload, ...],
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
) -> _ResidentTrace:
    model = DecoderOnlyTransformer(config.model).to(device)
    restore_pytorch_state(model, payloads)
    model.eval()
    input_device = input_ids.to(device)
    targets_device = targets.to(device)
    boundaries: dict[str, Tensor] = {}
    with torch.no_grad():
        positions = torch.arange(input_device.shape[1], device=device)
        hidden_states = model.token_embedding(input_device) + model.position_embedding(
            positions
        )
        hidden_states = model.embedding_dropout(hidden_states)
        boundaries["embedding"] = hidden_states.detach().cpu().contiguous()
        for index, block in enumerate(model.blocks):
            hidden_states = block(hidden_states)
            boundaries[f"block-{index}"] = hidden_states.detach().cpu().contiguous()
        normalized = model.final_norm(hidden_states)
        logits = model.lm_head(normalized)
        boundaries["final-head"] = logits.detach().cpu().contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets_device.reshape(-1)
        )
    synchronize_accelerator(device)
    result = _ResidentTrace(
        loss=float(loss.detach().cpu().item()),
        logits=logits.detach().cpu().contiguous(),
        boundaries=boundaries,
    )
    del model, hidden_states, normalized, logits, loss, input_device, targets_device
    _release_accelerator(device)
    return result


def _read_group(
    store: VersionedTensorStore,
    manifest_id: str,
    spec: ExecutionGroupSpec,
    records: dict[str, TensorRecord],
    device: torch.device,
) -> tuple[dict[str, Tensor], float, float, int, int]:
    read_started = time.perf_counter()
    payloads = tuple(
        store.read_tensor(records[name].tensor_id, manifest_id)
        for name in spec.tensor_names
    )
    read_seconds = time.perf_counter() - read_started
    materialization_started = time.perf_counter()
    tensors = {
        payload.logical_name: payload_to_torch(payload, device=device)
        for payload in payloads
    }
    synchronize_accelerator(device)
    materialization_seconds = time.perf_counter() - materialization_started
    logical_bytes = sum(records[name].byte_length for name in spec.tensor_names)
    referenced_chunks = sum(len(records[name].chunk_ids) for name in spec.tensor_names)
    return tensors, read_seconds, materialization_seconds, logical_bytes, referenced_chunks


def _embedding_forward(
    input_ids: Tensor,
    tensors: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    input_device = input_ids.to(device)
    positions = torch.arange(input_device.shape[1], device=device)
    return F.embedding(
        input_device, tensors["model.token_embedding.weight"]
    ) + F.embedding(positions, tensors["model.position_embedding.weight"])


def _block_forward(
    hidden_states: Tensor,
    tensors: dict[str, Tensor],
    config: ExperimentConfig,
    block_index: int,
) -> Tensor:
    prefix = f"model.blocks.{block_index}"
    normalized = F.layer_norm(
        hidden_states,
        (config.model.hidden_size,),
        tensors[f"{prefix}.attention_norm.weight"],
        tensors[f"{prefix}.attention_norm.bias"],
        config.model.norm_epsilon,
    )
    qkv = F.linear(normalized, tensors[f"{prefix}.attention.qkv.weight"])
    query, key, value = qkv.chunk(3, dim=-1)
    batch_size, sequence_length, hidden_size = hidden_states.shape
    head_size = hidden_size // config.model.heads

    def split_heads(value_tensor: Tensor) -> Tensor:
        return value_tensor.view(
            batch_size,
            sequence_length,
            config.model.heads,
            head_size,
        ).transpose(1, 2)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(head_size)
    causal_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=hidden_states.device,
    ).triu(diagonal=1)
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    probabilities = F.softmax(scores, dim=-1)
    context = probabilities @ value
    context = context.transpose(1, 2).contiguous().view(
        batch_size, sequence_length, hidden_size
    )
    hidden_states = hidden_states + F.linear(
        context, tensors[f"{prefix}.attention.output.weight"]
    )
    normalized = F.layer_norm(
        hidden_states,
        (config.model.hidden_size,),
        tensors[f"{prefix}.mlp_norm.weight"],
        tensors[f"{prefix}.mlp_norm.bias"],
        config.model.norm_epsilon,
    )
    intermediate = F.linear(normalized, tensors[f"{prefix}.mlp.input.weight"])
    intermediate = F.gelu(intermediate, approximate="tanh")
    return hidden_states + F.linear(
        intermediate, tensors[f"{prefix}.mlp.output.weight"]
    )


def _final_forward(
    hidden_states: Tensor,
    tensors: dict[str, Tensor],
    config: ExperimentConfig,
) -> Tensor:
    normalized = F.layer_norm(
        hidden_states,
        (config.model.hidden_size,),
        tensors["model.final_norm.weight"],
        tensors["model.final_norm.bias"],
        config.model.norm_epsilon,
    )
    head_name = (
        "model.lm_head.weight"
        if "model.lm_head.weight" in tensors
        else "model.token_embedding.weight"
    )
    return F.linear(normalized, tensors[head_name])


def run_bounded_forward(
    config: ExperimentConfig,
    *,
    store_path: str | Path,
    output_path: str | Path | None = None,
    device_override: str | None = None,
    parameter_working_set_bytes: int = 1024**2,
) -> BoundedForwardResult:
    """Compare a resident forward with parameter groups loaded from storage."""

    if config.model.dropout != 0.0:
        raise ValueError("bounded forward currently requires model.dropout=0")
    if parameter_working_set_bytes <= 0:
        raise ValueError("parameter_working_set_bytes must be greater than zero")
    store_destination = Path(store_path)
    if store_destination.exists():
        raise FileExistsError(f"bounded-forward requires a new store: {store_destination}")

    device = resolve_device(device_override or config.training.device)
    seed_everything(config.training.seed)
    bootstrap_model = DecoderOnlyTransformer(config.model)
    parameter_count = bootstrap_model.parameter_count
    bootstrap_payloads = export_pytorch_model(bootstrap_model)
    del bootstrap_model
    gc.collect()

    store = VersionedTensorStore.create(
        store_destination,
        limits=_store_limits(config),
    )
    transaction = store.begin_transaction(committed_step=0)
    transaction.put_many(bootstrap_payloads)
    commit = transaction.commit()
    manifest = commit.manifest
    records = _record_map(manifest.tensors)
    groups = build_execution_groups(config, set(records))
    maximum_group_bytes = max(
        sum(records[name].byte_length for name in group.tensor_names)
        for group in groups
    )
    if maximum_group_bytes > parameter_working_set_bytes:
        raise WorkingSetExceededError(
            "largest execution group requires "
            f"{maximum_group_bytes} bytes but the budget is "
            f"{parameter_working_set_bytes} bytes"
        )

    generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 1)
    input_ids, targets = make_synthetic_lm_batch(
        batch_size=config.training.micro_batch_size,
        sequence_length=config.training.sequence_length,
        vocab_size=config.model.vocab_size,
        generator=generator,
    )
    data_checksum = batch_checksum(input_ids, targets)
    resident = _resident_trace(
        config,
        bootstrap_payloads,
        input_ids,
        targets,
        device,
    )
    del bootstrap_payloads
    gc.collect()

    group_metrics: list[ExecutionGroupMetrics] = []
    hidden_states: Tensor | None = None
    bounded_logits: Tensor | None = None
    unique_tensor_names: set[str] = set()
    total_tensor_reads = 0
    total_chunk_reads = 0
    total_logical_bytes = 0

    with torch.no_grad():
        for spec in groups:
            group_bytes = sum(records[name].byte_length for name in spec.tensor_names)
            if group_bytes > parameter_working_set_bytes:
                raise WorkingSetExceededError(
                    f"execution group {spec.name} requires {group_bytes} bytes"
                )
            input_activation_bytes = (
                _activation_bytes(input_ids)
                if hidden_states is None
                else _activation_bytes(hidden_states)
            )
            tensors, read_seconds, materialization_seconds, logical_bytes, chunks = (
                _read_group(store, manifest.manifest_id, spec, records, device)
            )
            materialized_memory = accelerator_memory_metrics(device)
            compute_started = time.perf_counter()
            if spec.name == "embedding":
                output = _embedding_forward(input_ids, tensors, device)
            elif spec.name.startswith("block-"):
                if hidden_states is None:
                    raise RuntimeError("block execution requires hidden states")
                block_index = int(spec.name.split("-", maxsplit=1)[1])
                output = _block_forward(hidden_states, tensors, config, block_index)
            else:
                if hidden_states is None:
                    raise RuntimeError("final execution requires hidden states")
                output = _final_forward(hidden_states, tensors, config)
            synchronize_accelerator(device)
            compute_seconds = time.perf_counter() - compute_started
            compute_memory = accelerator_memory_metrics(device)
            output_cpu = output.detach().cpu().contiguous()
            comparison = compare_tensors(resident.boundaries[spec.name], output_cpu)
            output_checksum = tensor_checksum(output_cpu)
            output_activation_bytes = _activation_bytes(output)
            process_after_compute = process_rss_bytes()

            if spec.name == "final-head":
                bounded_logits = output
            else:
                hidden_states = output

            total_tensor_reads += len(spec.tensor_names)
            unique_tensor_names.update(spec.tensor_names)
            total_chunk_reads += chunks
            total_logical_bytes += logical_bytes
            del tensors, output_cpu
            release_started = time.perf_counter()
            _release_accelerator(device)
            release_seconds = time.perf_counter() - release_started
            released_memory = accelerator_memory_metrics(device)
            group_metrics.append(
                ExecutionGroupMetrics(
                    ordinal=spec.ordinal,
                    name=spec.name,
                    tensor_names=spec.tensor_names,
                    tensor_count=len(spec.tensor_names),
                    referenced_chunk_reads=chunks,
                    logical_parameter_bytes=logical_bytes,
                    read_seconds=read_seconds,
                    materialization_seconds=materialization_seconds,
                    compute_seconds=compute_seconds,
                    release_seconds=release_seconds,
                    input_activation_bytes=input_activation_bytes,
                    output_activation_bytes=output_activation_bytes,
                    output_checksum=output_checksum,
                    resident_comparison=comparison,
                    process_rss_after_compute_bytes=process_after_compute,
                    accelerator_after_materialization=materialized_memory,
                    accelerator_after_compute=compute_memory,
                    accelerator_after_release=released_memory,
                )
            )

    if bounded_logits is None:
        raise RuntimeError("bounded forward did not produce logits")
    targets_device = targets.to(device)
    bounded_loss_tensor = F.cross_entropy(
        bounded_logits.reshape(-1, bounded_logits.size(-1)), targets_device.reshape(-1)
    )
    synchronize_accelerator(device)
    bounded_loss = float(bounded_loss_tensor.detach().cpu().item())
    bounded_logits_cpu = bounded_logits.detach().cpu().contiguous()
    logits_comparison = compare_tensors(resident.logits, bounded_logits_cpu)
    manifest_after = store.current_manifest()
    verification = store.verify(manifest.manifest_id)
    result = BoundedForwardResult(
        schema_version=BOUNDED_FORWARD_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        store_path=str(store_destination),
        manifest_id=manifest.manifest_id,
        manifest_checksum=manifest.manifest_checksum,
        parameter_count=parameter_count,
        batch_checksum=data_checksum,
        parameter_working_set_budget_bytes=parameter_working_set_bytes,
        maximum_group_parameter_bytes=maximum_group_bytes,
        budget_respected=maximum_group_bytes <= parameter_working_set_bytes,
        bootstrap_model_released_before_bounded=True,
        resident_model_released_before_bounded=True,
        retained_activations_during_bounded=True,
        total_tensor_reads=total_tensor_reads,
        unique_tensor_reads=len(unique_tensor_names),
        repeated_tensor_reads=total_tensor_reads - len(unique_tensor_names),
        total_referenced_chunk_reads=total_chunk_reads,
        total_logical_parameter_bytes_read=total_logical_bytes,
        resident_loss=resident.loss,
        bounded_loss=bounded_loss,
        loss_absolute_difference=abs(resident.loss - bounded_loss),
        resident_logits_checksum=tensor_checksum(resident.logits),
        bounded_logits_checksum=tensor_checksum(bounded_logits_cpu),
        logits_comparison=logits_comparison,
        execution_groups=tuple(group_metrics),
        manifest_unchanged=(
            manifest_after.manifest_id == manifest.manifest_id
            and manifest_after.manifest_checksum == manifest.manifest_checksum
        ),
        store_verified_tensor_count=verification.tensor_count,
        store_verified_chunk_count=verification.chunk_count,
        store_verified_logical_bytes=verification.logical_bytes,
        store_verified_physical_bytes=verification.physical_bytes,
    )
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result
