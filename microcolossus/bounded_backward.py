"""Bounded group-by-group backward propagation backed by versioned stores."""

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

from .bounded_forward import (
    ExecutionGroupSpec,
    WorkingSetExceededError,
    _activation_bytes,
    _block_forward,
    _embedding_forward,
    _final_forward,
    _read_group,
    _record_map,
    _release_accelerator,
    _store_limits,
    build_execution_groups,
    tensor_checksum,
)
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .storage import StoreLimits, TensorKind, TensorPayload, VersionedTensorStore
from .storage.adapters import export_pytorch_model, payload_from_torch, payload_to_torch
from .storage.schema import StoreTelemetry, TensorRecord
from .storage_training import StateComparison, compare_states
from .telemetry import (
    AcceleratorMemoryMetrics,
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training import make_synthetic_lm_batch, resolve_device, seed_everything

BOUNDED_BACKWARD_SCHEMA_VERSION = "microcolossus.bounded-backward.v1"


class GradientWorkingSetExceededError(RuntimeError):
    """Raised when one gradient group exceeds the declared budget."""


@dataclass(frozen=True)
class ForwardBoundaryMetrics:
    ordinal: int
    name: str
    tensor_count: int
    logical_parameter_bytes: int
    referenced_chunk_reads: int
    read_seconds: float
    materialization_seconds: float
    compute_seconds: float
    release_seconds: float
    output_activation_bytes: int
    output_checksum: str
    process_rss_after_compute_bytes: int
    accelerator_after_compute: AcceleratorMemoryMetrics


@dataclass(frozen=True)
class BackwardGroupMetrics:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    logical_parameter_bytes: int
    referenced_chunk_reads: int
    input_activation_bytes: int
    output_activation_bytes: int
    incoming_activation_gradient_bytes: int
    outgoing_activation_gradient_bytes: int
    parameter_read_seconds: float
    materialization_seconds: float
    recomputation_seconds: float
    backward_seconds: float
    gradient_extraction_seconds: float
    gradient_commit_seconds: float
    release_seconds: float
    gradient_logical_bytes_written: int
    gradient_physical_bytes_written: int
    gradient_chunk_writes: int
    gradient_chunks_reused: int
    local_gradient_checksum: str
    upstream_gradient_checksum: str | None
    process_rss_after_backward_bytes: int
    accelerator_after_backward: AcceleratorMemoryMetrics
    accelerator_after_release: AcceleratorMemoryMetrics


@dataclass(frozen=True)
class ResidentGradientTrace:
    loss: float
    gradient_norm: float
    gradients: tuple[TensorPayload, ...]


@dataclass(frozen=True)
class BoundedBackwardResult:
    schema_version: str
    experiment: str
    device: str
    parameter_store_path: str
    oracle_gradient_store_path: str
    gradient_store_path: str
    parameter_manifest_id: str
    parameter_manifest_checksum: str
    oracle_gradient_manifest_id: str
    gradient_manifest_id: str
    oracle_gradient_store_commit: StoreTelemetry
    parameter_count: int
    batch_checksum: str
    parameter_working_set_budget_bytes: int
    gradient_working_set_budget_bytes: int
    maximum_parameter_group_bytes: int
    maximum_gradient_group_bytes: int
    parameter_budget_respected: bool
    gradient_budget_respected: bool
    retained_cpu_activations: bool
    retained_cpu_activation_bytes: int
    resident_oracle_released_before_bounded: bool
    bootstrap_payloads_released_before_bounded: bool
    full_gradient_state_materialized_for_validation: bool
    forward_groups: tuple[ForwardBoundaryMetrics, ...]
    backward_groups: tuple[BackwardGroupMetrics, ...]
    backward_group_order: tuple[str, ...]
    resident_loss: float
    bounded_loss: float
    loss_absolute_difference: float
    resident_gradient_norm: float
    oracle_store_gradient_norm: float
    oracle_store_norm_absolute_difference: float
    bounded_gradient_norm: float
    gradient_norm_absolute_difference: float
    future_clip_coefficient: float
    gradient_tensor_count: int
    total_parameter_tensor_reads: int
    total_parameter_chunk_reads: int
    total_parameter_logical_bytes_read: int
    total_gradient_logical_bytes_written: int
    total_gradient_physical_bytes_written: int
    total_gradient_chunk_writes: int
    total_gradient_chunks_reused: int
    tied_gradient_accumulation_count: int
    tied_gradient_version: int
    resident_vs_bounded_gradients: StateComparison
    parameter_manifest_unchanged: bool
    parameter_store_verified_tensor_count: int
    parameter_store_verified_chunk_count: int
    oracle_gradient_store_verified_tensor_count: int
    oracle_gradient_store_verified_chunk_count: int
    gradient_store_verified_tensor_count: int
    gradient_store_verified_chunk_count: int
    gradient_versions: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gradient_logical_name(parameter_name: str) -> str:
    return f"gradient.{parameter_name}"


def _gradient_metadata(parameter_name: str) -> tuple[tuple[str, str], ...]:
    return (
        ("backend", "pytorch"),
        ("parameter_name", parameter_name),
    )


def _batch_checksum(input_ids: Tensor, targets: Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in (("input_ids", input_ids), ("targets", targets)):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        byte_view = contiguous.reshape(-1).view(torch.uint8)
        digest.update(bytes(cast(Iterable[int], byte_view.untyped_storage())))
    return digest.hexdigest()


def _gradient_payload(parameter_name: str, gradient: Tensor) -> TensorPayload:
    return payload_from_torch(
        gradient.detach().cpu().contiguous(),
        logical_name=_gradient_logical_name(parameter_name),
        kind=TensorKind.GRADIENT,
        metadata=_gradient_metadata(parameter_name),
    )


def _gradient_map(records: tuple[TensorRecord, ...]) -> dict[str, TensorRecord]:
    result: dict[str, TensorRecord] = {}
    for record in records:
        parameter_name = dict(record.metadata).get("parameter_name")
        if parameter_name is None:
            continue
        if parameter_name in result:
            raise ValueError(f"duplicate gradient for parameter {parameter_name}")
        result[parameter_name] = record
    return result


def _payload_digest(payloads: tuple[TensorPayload, ...]) -> str:
    digest = hashlib.sha256()
    for payload in sorted(payloads, key=lambda item: item.logical_name):
        digest.update(payload.logical_name.encode("utf-8"))
        digest.update(payload.checksum.encode("ascii"))
    return digest.hexdigest()


def _enable_parameter_gradients(tensors: dict[str, Tensor]) -> None:
    for tensor in tensors.values():
        tensor.requires_grad_(True)


def _extract_group_gradients(tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    gradients: dict[str, Tensor] = {}
    for name, tensor in tensors.items():
        if tensor.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradients[name] = tensor.grad.detach().cpu().contiguous()
    return gradients


def _resident_gradient_trace(
    config: ExperimentConfig,
    payloads: tuple[TensorPayload, ...],
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
) -> ResidentGradientTrace:
    from .storage.adapters import restore_pytorch_model

    model = DecoderOnlyTransformer(config.model).to(device)
    restore_pytorch_model(model, payloads)
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(input_ids.to(device), targets.to(device))
    if output.loss is None:
        raise RuntimeError("resident model did not return a loss")
    output.loss.backward()
    synchronize_accelerator(device)

    squared_norms: list[float] = []
    gradients: list[TensorPayload] = []
    for name, parameter in model.named_parameters(remove_duplicate=True):
        if parameter.grad is None:
            raise RuntimeError(f"resident gradient missing for {name}")
        squared_sum = parameter.grad.detach().float().pow(2).sum()
        squared_norms.append(float(squared_sum.cpu().item()))
        gradients.append(_gradient_payload(f"model.{name}", parameter.grad))
    trace = ResidentGradientTrace(
        loss=float(output.loss.detach().cpu().item()),
        gradient_norm=math.sqrt(math.fsum(squared_norms)),
        gradients=tuple(sorted(gradients, key=lambda item: item.logical_name)),
    )
    del model, output
    _release_accelerator(device)
    return trace


def _bounded_forward_activations(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, TensorRecord],
    groups: tuple[ExecutionGroupSpec, ...],
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
) -> tuple[dict[str, Tensor], float, tuple[ForwardBoundaryMetrics, ...]]:
    activations: dict[str, Tensor] = {}
    metrics: list[ForwardBoundaryMetrics] = []
    hidden_states: Tensor | None = None
    bounded_loss: float | None = None

    with torch.no_grad():
        for spec in groups:
            tensors, read_seconds, materialization_seconds, logical_bytes, chunk_reads = (
                _read_group(store, manifest_id, spec, records, device)
            )
            compute_started = time.perf_counter()
            if spec.name == "embedding":
                output = _embedding_forward(input_ids, tensors, device)
            elif spec.name.startswith("block-"):
                if hidden_states is None:
                    raise RuntimeError("block executed before embeddings")
                output = _block_forward(
                    hidden_states,
                    tensors,
                    config,
                    int(spec.name.split("-", maxsplit=1)[1]),
                )
            else:
                if hidden_states is None:
                    raise RuntimeError("final head executed before hidden states")
                output = _final_forward(hidden_states, tensors, config)
                loss = F.cross_entropy(
                    output.reshape(-1, output.size(-1)),
                    targets.to(device).reshape(-1),
                )
                bounded_loss = float(loss.detach().cpu().item())
            synchronize_accelerator(device)
            compute_seconds = time.perf_counter() - compute_started
            memory = accelerator_memory_metrics(device)
            output_cpu = output.detach().cpu().contiguous()
            if spec.name != "final-head":
                activations[spec.name] = output_cpu
                hidden_states = output
            release_started = time.perf_counter()
            del tensors
            gc.collect()
            synchronize_accelerator(device)
            release_seconds = time.perf_counter() - release_started
            metrics.append(
                ForwardBoundaryMetrics(
                    ordinal=spec.ordinal,
                    name=spec.name,
                    tensor_count=len(spec.tensor_names),
                    logical_parameter_bytes=logical_bytes,
                    referenced_chunk_reads=chunk_reads,
                    read_seconds=read_seconds,
                    materialization_seconds=materialization_seconds,
                    compute_seconds=compute_seconds,
                    release_seconds=release_seconds,
                    output_activation_bytes=_activation_bytes(output_cpu),
                    output_checksum=tensor_checksum(output_cpu),
                    process_rss_after_compute_bytes=process_rss_bytes(),
                    accelerator_after_compute=memory,
                )
            )
            if spec.name == "final-head":
                del output, output_cpu
            else:
                del output_cpu

    if bounded_loss is None:
        raise RuntimeError("bounded forward did not produce a loss")
    del hidden_states
    _release_accelerator(device)
    return activations, bounded_loss, tuple(metrics)


def _group_input_activation(
    spec: ExecutionGroupSpec,
    groups: tuple[ExecutionGroupSpec, ...],
    activations: dict[str, Tensor],
) -> Tensor:
    if spec.name == "final-head":
        previous = groups[-2].name
        return activations[previous]
    if spec.name.startswith("block-"):
        index = int(spec.name.split("-", maxsplit=1)[1])
        previous = "embedding" if index == 0 else f"block-{index - 1}"
        return activations[previous]
    raise ValueError(f"group {spec.name} does not consume a hidden activation")


def _existing_gradient(
    store: VersionedTensorStore,
    parameter_name: str,
) -> Tensor | None:
    records = _gradient_map(store.current_manifest().tensors)
    record = records.get(parameter_name)
    if record is None:
        return None
    return payload_to_torch(store.read_tensor(record.tensor_id)).detach().cpu().contiguous()


def _commit_group_gradients(
    store: VersionedTensorStore,
    gradients: dict[str, Tensor],
) -> tuple[StoreTelemetry, tuple[TensorPayload, ...]]:
    payloads: list[TensorPayload] = []
    for parameter_name, contribution in sorted(gradients.items()):
        existing = _existing_gradient(store, parameter_name)
        combined = contribution if existing is None else existing + contribution
        payloads.append(_gradient_payload(parameter_name, combined))
    transaction = store.begin_transaction(
        committed_step=store.current_manifest().committed_step + 1
    )
    transaction.put_many(payloads)
    result = transaction.commit()
    return result.telemetry, tuple(payloads)


def _stream_gradient_norm(store: VersionedTensorStore) -> float:
    squared_norms: list[float] = []
    for record in sorted(store.current_manifest().tensors, key=lambda item: item.logical_name):
        payload = store.read_tensor(record.tensor_id)
        value = payload_to_torch(payload).float()
        squared_norms.append(float(value.pow(2).sum().item()))
        del value, payload
    return math.sqrt(math.fsum(squared_norms))


def _read_all_gradients(store: VersionedTensorStore) -> tuple[TensorPayload, ...]:
    return tuple(
        store.read_tensor(record.tensor_id)
        for record in sorted(
            store.current_manifest().tensors,
            key=lambda item: item.logical_name,
        )
    )


def run_bounded_backward(
    config: ExperimentConfig,
    *,
    parameter_store_path: str | Path,
    oracle_gradient_store_path: str | Path,
    gradient_store_path: str | Path,
    output_path: str | Path | None = None,
    device_override: str | None = None,
    parameter_working_set_bytes: int = 1024**2,
    gradient_working_set_bytes: int = 1024**2,
) -> BoundedBackwardResult:
    """Compare resident gradients with group-bounded backward propagation."""

    if config.model.dropout != 0.0:
        raise ValueError("bounded backward currently requires model.dropout=0")
    if parameter_working_set_bytes <= 0:
        raise ValueError("parameter_working_set_bytes must be greater than zero")
    if gradient_working_set_bytes <= 0:
        raise ValueError("gradient_working_set_bytes must be greater than zero")
    parameter_destination = Path(parameter_store_path)
    oracle_gradient_destination = Path(oracle_gradient_store_path)
    gradient_destination = Path(gradient_store_path)
    if parameter_destination.exists():
        raise FileExistsError(
            f"bounded-backward requires a new parameter store: {parameter_destination}"
        )
    if oracle_gradient_destination.exists():
        raise FileExistsError(
            "bounded-backward requires a new oracle gradient store: "
            f"{oracle_gradient_destination}"
        )
    if gradient_destination.exists():
        raise FileExistsError(
            f"bounded-backward requires a new gradient store: {gradient_destination}"
        )

    device = resolve_device(device_override or config.training.device)
    seed_everything(config.training.seed)
    bootstrap_model = DecoderOnlyTransformer(config.model)
    parameter_count = bootstrap_model.parameter_count
    bootstrap_payloads = export_pytorch_model(bootstrap_model)
    del bootstrap_model
    gc.collect()

    parameter_store = VersionedTensorStore.create(
        parameter_destination,
        limits=_store_limits(config),
    )
    parameter_transaction = parameter_store.begin_transaction(committed_step=0)
    parameter_transaction.put_many(bootstrap_payloads)
    parameter_commit = parameter_transaction.commit()
    parameter_manifest = parameter_commit.manifest
    records = _record_map(parameter_manifest.tensors)
    groups = build_execution_groups(config, set(records))
    maximum_parameter_group_bytes = max(
        sum(records[name].byte_length for name in group.tensor_names) for group in groups
    )
    if maximum_parameter_group_bytes > parameter_working_set_bytes:
        raise WorkingSetExceededError(
            "largest execution group requires "
            f"{maximum_parameter_group_bytes} bytes but the parameter budget is "
            f"{parameter_working_set_bytes} bytes"
        )
    maximum_gradient_group_bytes = maximum_parameter_group_bytes
    if maximum_gradient_group_bytes > gradient_working_set_bytes:
        raise GradientWorkingSetExceededError(
            "largest gradient group requires "
            f"{maximum_gradient_group_bytes} bytes but the gradient budget is "
            f"{gradient_working_set_bytes} bytes"
        )

    generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 1)
    input_ids, targets = make_synthetic_lm_batch(
        batch_size=config.training.micro_batch_size,
        sequence_length=config.training.sequence_length,
        vocab_size=config.model.vocab_size,
        generator=generator,
    )
    data_checksum = _batch_checksum(input_ids, targets)
    resident = _resident_gradient_trace(
        config,
        bootstrap_payloads,
        input_ids,
        targets,
        device,
    )
    oracle_gradient_store = VersionedTensorStore.create(
        oracle_gradient_destination,
        limits=StoreLimits(
            chunk_size_bytes=parameter_store.limits.chunk_size_bytes,
            max_storage_bytes=parameter_store.limits.max_storage_bytes,
            max_staging_bytes=parameter_store.limits.max_staging_bytes,
        ),
    )
    oracle_transaction = oracle_gradient_store.begin_transaction(committed_step=0)
    oracle_transaction.put_many(resident.gradients)
    oracle_commit = oracle_transaction.commit()
    resident_loss = resident.loss
    resident_gradient_norm = resident.gradient_norm
    del resident, bootstrap_payloads
    gc.collect()

    activations, bounded_forward_loss, forward_metrics = _bounded_forward_activations(
        config,
        parameter_store,
        parameter_manifest.manifest_id,
        records,
        groups,
        input_ids,
        targets,
        device,
    )

    gradient_store = VersionedTensorStore.create(
        gradient_destination,
        limits=StoreLimits(
            chunk_size_bytes=parameter_store.limits.chunk_size_bytes,
            max_storage_bytes=parameter_store.limits.max_storage_bytes,
            max_staging_bytes=parameter_store.limits.max_staging_bytes,
        ),
    )

    upstream: Tensor | None = None
    bounded_loss = bounded_forward_loss
    backward_metrics: list[BackwardGroupMetrics] = []
    tied_accumulations = 0
    for reverse_ordinal, spec in enumerate(reversed(groups)):
        tensors, read_seconds, materialization_seconds, logical_bytes, chunk_reads = (
            _read_group(
                parameter_store,
                parameter_manifest.manifest_id,
                spec,
                records,
                device,
            )
        )
        if logical_bytes > parameter_working_set_bytes:
            raise WorkingSetExceededError(
                f"execution group {spec.name} requires {logical_bytes} parameter bytes"
            )
        if logical_bytes > gradient_working_set_bytes:
            raise GradientWorkingSetExceededError(
                f"execution group {spec.name} requires {logical_bytes} gradient bytes"
            )
        _enable_parameter_gradients(tensors)
        input_activation_bytes = 0
        output_activation_bytes = 0
        incoming_gradient_bytes = 0 if upstream is None else _activation_bytes(upstream)
        recompute_started = time.perf_counter()
        if spec.name == "final-head":
            local_input = _group_input_activation(spec, groups, activations).to(device)
            local_input.requires_grad_(True)
            output = _final_forward(local_input, tensors, config)
            loss = F.cross_entropy(
                output.reshape(-1, output.size(-1)),
                targets.to(device).reshape(-1),
            )
            synchronize_accelerator(device)
            recomputation_seconds = time.perf_counter() - recompute_started
            backward_started = time.perf_counter()
            loss.backward()
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            bounded_loss = float(loss.detach().cpu().item())
            output_activation_bytes = _activation_bytes(output)
            del loss
        elif spec.name.startswith("block-"):
            local_input = _group_input_activation(spec, groups, activations).to(device)
            local_input.requires_grad_(True)
            output = _block_forward(
                local_input,
                tensors,
                config,
                int(spec.name.split("-", maxsplit=1)[1]),
            )
            synchronize_accelerator(device)
            recomputation_seconds = time.perf_counter() - recompute_started
            if upstream is None:
                raise RuntimeError("block backward requires an upstream gradient")
            backward_started = time.perf_counter()
            output.backward(upstream.to(device))
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            output_activation_bytes = _activation_bytes(output)
        else:
            local_input = input_ids
            output = _embedding_forward(input_ids, tensors, device)
            synchronize_accelerator(device)
            recomputation_seconds = time.perf_counter() - recompute_started
            if upstream is None:
                raise RuntimeError("embedding backward requires an upstream gradient")
            backward_started = time.perf_counter()
            output.backward(upstream.to(device))
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            output_activation_bytes = _activation_bytes(output)

        if spec.name != "embedding":
            input_activation_bytes = _activation_bytes(local_input)
            if local_input.grad is None:
                raise RuntimeError(f"missing upstream gradient after {spec.name}")
            next_upstream = local_input.grad.detach().cpu().contiguous()
        else:
            next_upstream = None

        extraction_started = time.perf_counter()
        gradients = _extract_group_gradients(tensors)
        gradient_payloads = tuple(
            _gradient_payload(name, gradient) for name, gradient in sorted(gradients.items())
        )
        gradient_extraction_seconds = time.perf_counter() - extraction_started
        local_gradient_checksum = _payload_digest(gradient_payloads)
        if "model.token_embedding.weight" in gradients:
            tied_accumulations += 1

        commit_started = time.perf_counter()
        commit_telemetry, committed_payloads = _commit_group_gradients(
            gradient_store,
            gradients,
        )
        gradient_commit_seconds = time.perf_counter() - commit_started
        del committed_payloads
        memory = accelerator_memory_metrics(device)
        process_rss = process_rss_bytes()
        release_started = time.perf_counter()
        del tensors, gradients, gradient_payloads, output
        if spec.name != "embedding":
            del local_input
        gc.collect()
        synchronize_accelerator(device)
        release_seconds = time.perf_counter() - release_started
        release_memory = accelerator_memory_metrics(device)
        upstream = next_upstream
        backward_metrics.append(
            BackwardGroupMetrics(
                ordinal=reverse_ordinal,
                name=spec.name,
                tensor_names=spec.tensor_names,
                logical_parameter_bytes=logical_bytes,
                referenced_chunk_reads=chunk_reads,
                input_activation_bytes=input_activation_bytes,
                output_activation_bytes=output_activation_bytes,
                incoming_activation_gradient_bytes=incoming_gradient_bytes,
                outgoing_activation_gradient_bytes=(
                    0 if next_upstream is None else _activation_bytes(next_upstream)
                ),
                parameter_read_seconds=read_seconds,
                materialization_seconds=materialization_seconds,
                recomputation_seconds=recomputation_seconds,
                backward_seconds=backward_seconds,
                gradient_extraction_seconds=gradient_extraction_seconds,
                gradient_commit_seconds=gradient_commit_seconds,
                release_seconds=release_seconds,
                gradient_logical_bytes_written=commit_telemetry.logical_bytes_written,
                gradient_physical_bytes_written=commit_telemetry.physical_bytes_written,
                gradient_chunk_writes=commit_telemetry.chunk_writes,
                gradient_chunks_reused=commit_telemetry.chunks_reused,
                local_gradient_checksum=local_gradient_checksum,
                upstream_gradient_checksum=(
                    None if next_upstream is None else tensor_checksum(next_upstream)
                ),
                process_rss_after_backward_bytes=process_rss,
                accelerator_after_backward=memory,
                accelerator_after_release=release_memory,
            )
        )

    oracle_store_gradient_norm = _stream_gradient_norm(oracle_gradient_store)
    bounded_gradient_norm = _stream_gradient_norm(gradient_store)
    oracle_gradients = _read_all_gradients(oracle_gradient_store)
    bounded_gradients = _read_all_gradients(gradient_store)
    resident_vs_bounded = compare_states(oracle_gradients, bounded_gradients)
    gradient_manifest = gradient_store.current_manifest()
    tied_record = _gradient_map(gradient_manifest.tensors)["model.token_embedding.weight"]
    parameter_after = parameter_store.current_manifest()
    parameter_verification = parameter_store.verify()
    oracle_gradient_verification = oracle_gradient_store.verify()
    gradient_verification = gradient_store.verify()
    max_norm = config.training.gradient_clip_norm
    future_clip_coefficient = (
        1.0
        if max_norm is None
        else min(1.0, max_norm / (bounded_gradient_norm + 1e-6))
    )
    retained_activation_bytes = sum(_activation_bytes(value) for value in activations.values())
    total_gradient_logical = sum(
        item.gradient_logical_bytes_written for item in backward_metrics
    )
    total_gradient_physical = sum(
        item.gradient_physical_bytes_written for item in backward_metrics
    )
    result = BoundedBackwardResult(
        schema_version=BOUNDED_BACKWARD_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        parameter_store_path=str(parameter_destination),
        oracle_gradient_store_path=str(oracle_gradient_destination),
        gradient_store_path=str(gradient_destination),
        parameter_manifest_id=parameter_manifest.manifest_id,
        parameter_manifest_checksum=parameter_manifest.manifest_checksum,
        oracle_gradient_manifest_id=oracle_commit.manifest.manifest_id,
        gradient_manifest_id=gradient_manifest.manifest_id,
        oracle_gradient_store_commit=oracle_commit.telemetry,
        parameter_count=parameter_count,
        batch_checksum=data_checksum,
        parameter_working_set_budget_bytes=parameter_working_set_bytes,
        gradient_working_set_budget_bytes=gradient_working_set_bytes,
        maximum_parameter_group_bytes=maximum_parameter_group_bytes,
        maximum_gradient_group_bytes=maximum_gradient_group_bytes,
        parameter_budget_respected=maximum_parameter_group_bytes <= parameter_working_set_bytes,
        gradient_budget_respected=maximum_gradient_group_bytes <= gradient_working_set_bytes,
        retained_cpu_activations=True,
        retained_cpu_activation_bytes=retained_activation_bytes,
        resident_oracle_released_before_bounded=True,
        bootstrap_payloads_released_before_bounded=True,
        full_gradient_state_materialized_for_validation=True,
        forward_groups=forward_metrics,
        backward_groups=tuple(backward_metrics),
        backward_group_order=tuple(item.name for item in backward_metrics),
        resident_loss=resident_loss,
        bounded_loss=bounded_loss,
        loss_absolute_difference=abs(resident_loss - bounded_loss),
        resident_gradient_norm=resident_gradient_norm,
        oracle_store_gradient_norm=oracle_store_gradient_norm,
        oracle_store_norm_absolute_difference=abs(
            resident_gradient_norm - oracle_store_gradient_norm
        ),
        bounded_gradient_norm=bounded_gradient_norm,
        gradient_norm_absolute_difference=abs(
            resident_gradient_norm - bounded_gradient_norm
        ),
        future_clip_coefficient=future_clip_coefficient,
        gradient_tensor_count=len(gradient_manifest.tensors),
        total_parameter_tensor_reads=(
            sum(item.tensor_count for item in forward_metrics)
            + sum(len(item.tensor_names) for item in backward_metrics)
        ),
        total_parameter_chunk_reads=(
            sum(item.referenced_chunk_reads for item in forward_metrics)
            + sum(item.referenced_chunk_reads for item in backward_metrics)
        ),
        total_parameter_logical_bytes_read=(
            sum(item.logical_parameter_bytes for item in forward_metrics)
            + sum(item.logical_parameter_bytes for item in backward_metrics)
        ),
        total_gradient_logical_bytes_written=total_gradient_logical,
        total_gradient_physical_bytes_written=total_gradient_physical,
        total_gradient_chunk_writes=sum(
            item.gradient_chunk_writes for item in backward_metrics
        ),
        total_gradient_chunks_reused=sum(
            item.gradient_chunks_reused for item in backward_metrics
        ),
        tied_gradient_accumulation_count=tied_accumulations,
        tied_gradient_version=tied_record.version,
        resident_vs_bounded_gradients=resident_vs_bounded,
        parameter_manifest_unchanged=(
            parameter_after.manifest_id == parameter_manifest.manifest_id
            and parameter_after.manifest_checksum == parameter_manifest.manifest_checksum
        ),
        parameter_store_verified_tensor_count=parameter_verification.tensor_count,
        parameter_store_verified_chunk_count=parameter_verification.chunk_count,
        oracle_gradient_store_verified_tensor_count=(
            oracle_gradient_verification.tensor_count
        ),
        oracle_gradient_store_verified_chunk_count=(
            oracle_gradient_verification.chunk_count
        ),
        gradient_store_verified_tensor_count=gradient_verification.tensor_count,
        gradient_store_verified_chunk_count=gradient_verification.chunk_count,
        gradient_versions=tuple(
            sorted((record.tensor_id, record.version) for record in gradient_manifest.tensors)
        ),
    )
    if output_path is not None:
        write_json_atomic(output_path, result)
    del activations, oracle_gradients, bounded_gradients
    _release_accelerator(device)
    return result
