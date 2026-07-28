"""Synchronous activation-recomputation reference for bounded backward execution.

The established bounded backward path retains every forward boundary on CPU.
This module provides the first M6B reference path: the forward retains no
boundary activation, and each reverse group deterministically replays the
prefix required to reconstruct its local input.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .bounded_backward import (
    GradientWorkingSetExceededError,
    _batch_checksum,
    _commit_group_gradients,
    _enable_parameter_gradients,
    _extract_group_gradients,
    _gradient_map,
    _gradient_payload,
    _payload_digest,
    _read_all_gradients,
    _resident_gradient_trace,
    _stream_gradient_norm,
)
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
from .storage import StoreLimits, VersionedTensorStore
from .storage.adapters import export_pytorch_model
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

ACTIVATION_RECOMPUTE_SCHEMA_VERSION = "microcolossus.activation-recompute.v1"


class ActivationWorkingSetExceededError(RuntimeError):
    """Raised when a retained activation working set exceeds its budget."""


class WorkspaceWorkingSetExceededError(RuntimeError):
    """Raised when a local recomputation workspace exceeds its budget."""


@dataclass(frozen=True)
class ActivationForwardGroupMetrics:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    tensor_count: int
    logical_parameter_bytes: int
    referenced_chunk_reads: int
    parameter_read_seconds: float
    materialization_seconds: float
    compute_seconds: float
    release_seconds: float
    input_activation_bytes: int
    output_activation_bytes: int
    logical_workspace_bytes: int
    output_checksum: str
    retained_after_group: bool
    process_rss_after_compute_bytes: int
    accelerator_after_compute: AcceleratorMemoryMetrics


@dataclass(frozen=True)
class PrefixReplayMetrics:
    target_group: str
    replayed_group_names: tuple[str, ...]
    parameter_tensor_reads: int
    parameter_chunk_reads: int
    parameter_logical_bytes_read: int
    parameter_read_seconds: float
    materialization_seconds: float
    compute_seconds: float
    release_seconds: float
    maximum_workspace_bytes: int
    output_activation_bytes: int
    output_checksum: str


@dataclass(frozen=True)
class ActivationBackwardGroupMetrics:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    prefix_replay: PrefixReplayMetrics | None
    logical_parameter_bytes: int
    referenced_chunk_reads: int
    input_activation_bytes: int
    output_activation_bytes: int
    incoming_activation_gradient_bytes: int
    outgoing_activation_gradient_bytes: int
    logical_workspace_bytes: int
    parameter_read_seconds: float
    materialization_seconds: float
    local_forward_seconds: float
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
class ActivationRecomputeResult:
    schema_version: str
    experiment: str
    device: str
    activation_policy: str
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
    activation_working_set_budget_bytes: int
    workspace_working_set_budget_bytes: int
    maximum_parameter_group_bytes: int
    maximum_gradient_group_bytes: int
    maximum_retained_activation_bytes: int
    maximum_workspace_bytes: int
    parameter_budget_respected: bool
    gradient_budget_respected: bool
    activation_budget_respected: bool
    workspace_budget_respected: bool
    retained_forward_boundary_count: int
    retained_forward_boundary_bytes: int
    forward_groups: tuple[ActivationForwardGroupMetrics, ...]
    backward_groups: tuple[ActivationBackwardGroupMetrics, ...]
    backward_group_order: tuple[str, ...]
    total_prefix_replayed_groups: int
    total_prefix_parameter_tensor_reads: int
    total_prefix_parameter_chunk_reads: int
    total_prefix_parameter_logical_bytes_read: int
    total_prefix_recomputation_seconds: float
    resident_loss: float
    recomputed_loss: float
    loss_absolute_difference: float
    resident_gradient_norm: float
    oracle_store_gradient_norm: float
    oracle_store_norm_absolute_difference: float
    recomputed_gradient_norm: float
    gradient_norm_absolute_difference: float
    future_clip_coefficient: float
    gradient_tensor_count: int
    tied_gradient_accumulation_count: int
    tied_gradient_version: int
    resident_vs_recomputed_gradients: StateComparison
    parameter_manifest_unchanged: bool
    parameter_store_verified_tensor_count: int
    oracle_gradient_store_verified_tensor_count: int
    recomputed_gradient_store_verified_tensor_count: int
    full_gradient_state_materialized_for_validation: bool = True
    resident_oracle_materialized_for_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_limits(store: VersionedTensorStore) -> StoreLimits:
    return StoreLimits(
        chunk_size_bytes=store.limits.chunk_size_bytes,
        max_storage_bytes=store.limits.max_storage_bytes,
        max_staging_bytes=store.limits.max_staging_bytes,
    )


def _check_workspace(required: int, budget: int, context: str) -> None:
    if required > budget:
        raise WorkspaceWorkingSetExceededError(
            f"{context} requires {required} workspace bytes but the budget is {budget} bytes"
        )


def _check_activation(required: int, budget: int, context: str) -> None:
    if required > budget:
        raise ActivationWorkingSetExceededError(
            f"{context} requires {required} retained activation bytes but the budget is "
            f"{budget} bytes"
        )


def _forward_without_retained_boundaries(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, TensorRecord],
    groups: tuple[ExecutionGroupSpec, ...],
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    *,
    parameter_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[float, tuple[ActivationForwardGroupMetrics, ...], int]:
    """Execute a group forward while retaining no later-backward boundary."""

    hidden_states: Tensor | None = None
    metrics: list[ActivationForwardGroupMetrics] = []
    bounded_loss: float | None = None
    maximum_workspace = 0

    with torch.no_grad():
        for spec in groups:
            input_activation_bytes = (
                _activation_bytes(input_ids)
                if hidden_states is None
                else _activation_bytes(hidden_states)
            )
            tensors, read_seconds, materialization_seconds, logical_bytes, chunks = (
                _read_group(store, manifest_id, spec, records, device)
            )
            if logical_bytes > parameter_working_set_bytes:
                raise WorkingSetExceededError(
                    f"execution group {spec.name} requires {logical_bytes} parameter bytes"
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
                del loss
            synchronize_accelerator(device)
            compute_seconds = time.perf_counter() - compute_started
            output_activation_bytes = _activation_bytes(output)
            workspace_bytes = input_activation_bytes + output_activation_bytes
            _check_workspace(
                workspace_bytes,
                workspace_working_set_bytes,
                f"forward group {spec.name}",
            )
            maximum_workspace = max(maximum_workspace, workspace_bytes)
            output_cpu = output.detach().cpu().contiguous()
            output_checksum = tensor_checksum(output_cpu)
            memory = accelerator_memory_metrics(device)
            process_rss = process_rss_bytes()

            release_started = time.perf_counter()
            del tensors, output_cpu
            if spec.name == "final-head":
                previous_hidden = hidden_states
                hidden_states = None
                if previous_hidden is not None:
                    del previous_hidden
                del output
            else:
                previous_hidden = hidden_states
                hidden_states = output
                if previous_hidden is not None:
                    del previous_hidden
            gc.collect()
            synchronize_accelerator(device)
            release_seconds = time.perf_counter() - release_started
            metrics.append(
                ActivationForwardGroupMetrics(
                    ordinal=spec.ordinal,
                    name=spec.name,
                    tensor_names=spec.tensor_names,
                    tensor_count=len(spec.tensor_names),
                    logical_parameter_bytes=logical_bytes,
                    referenced_chunk_reads=chunks,
                    parameter_read_seconds=read_seconds,
                    materialization_seconds=materialization_seconds,
                    compute_seconds=compute_seconds,
                    release_seconds=release_seconds,
                    input_activation_bytes=input_activation_bytes,
                    output_activation_bytes=output_activation_bytes,
                    logical_workspace_bytes=workspace_bytes,
                    output_checksum=output_checksum,
                    retained_after_group=False,
                    process_rss_after_compute_bytes=process_rss,
                    accelerator_after_compute=memory,
                )
            )

    if bounded_loss is None:
        raise RuntimeError("activation-recompute forward did not produce a loss")
    _release_accelerator(device)
    return bounded_loss, tuple(metrics), maximum_workspace


def _recompute_group_input(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, TensorRecord],
    groups: tuple[ExecutionGroupSpec, ...],
    target: ExecutionGroupSpec,
    input_ids: Tensor,
    device: torch.device,
    *,
    parameter_working_set_bytes: int,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[Tensor, PrefixReplayMetrics]:
    """Replay the deterministic prefix required to reconstruct one group input."""

    if target.ordinal <= 0:
        raise ValueError("embedding does not consume a hidden activation")
    hidden_states: Tensor | None = None
    replayed_names: list[str] = []
    tensor_reads = 0
    chunk_reads = 0
    logical_bytes_read = 0
    read_seconds_total = 0.0
    materialization_seconds_total = 0.0
    compute_seconds_total = 0.0
    release_seconds_total = 0.0
    maximum_workspace = 0

    with torch.no_grad():
        for spec in groups[: target.ordinal]:
            input_activation_bytes = (
                _activation_bytes(input_ids)
                if hidden_states is None
                else _activation_bytes(hidden_states)
            )
            tensors, read_seconds, materialization_seconds, logical_bytes, chunks = (
                _read_group(store, manifest_id, spec, records, device)
            )
            if logical_bytes > parameter_working_set_bytes:
                raise WorkingSetExceededError(
                    f"replayed group {spec.name} requires {logical_bytes} parameter bytes"
                )
            compute_started = time.perf_counter()
            if spec.name == "embedding":
                output = _embedding_forward(input_ids, tensors, device)
            elif spec.name.startswith("block-"):
                if hidden_states is None:
                    raise RuntimeError("replayed block executed before embeddings")
                output = _block_forward(
                    hidden_states,
                    tensors,
                    config,
                    int(spec.name.split("-", maxsplit=1)[1]),
                )
            else:
                raise RuntimeError("final head cannot be part of a prefix replay")
            synchronize_accelerator(device)
            compute_seconds = time.perf_counter() - compute_started
            output_activation_bytes = _activation_bytes(output)
            workspace_bytes = input_activation_bytes + output_activation_bytes
            _check_workspace(
                workspace_bytes,
                workspace_working_set_bytes,
                f"prefix replay group {spec.name}",
            )
            maximum_workspace = max(maximum_workspace, workspace_bytes)

            release_started = time.perf_counter()
            del tensors
            previous_hidden = hidden_states
            hidden_states = output
            if previous_hidden is not None:
                del previous_hidden
            gc.collect()
            synchronize_accelerator(device)
            release_seconds_total += time.perf_counter() - release_started

            replayed_names.append(spec.name)
            tensor_reads += len(spec.tensor_names)
            chunk_reads += chunks
            logical_bytes_read += logical_bytes
            read_seconds_total += read_seconds
            materialization_seconds_total += materialization_seconds
            compute_seconds_total += compute_seconds

    if hidden_states is None:
        raise RuntimeError(f"prefix replay for {target.name} produced no activation")
    output_cpu = hidden_states.detach().cpu().contiguous()
    output_activation_bytes = _activation_bytes(output_cpu)
    _check_activation(
        output_activation_bytes,
        activation_working_set_bytes,
        f"recomputed input for {target.name}",
    )
    checksum = tensor_checksum(output_cpu)
    del hidden_states
    _release_accelerator(device)
    return output_cpu, PrefixReplayMetrics(
        target_group=target.name,
        replayed_group_names=tuple(replayed_names),
        parameter_tensor_reads=tensor_reads,
        parameter_chunk_reads=chunk_reads,
        parameter_logical_bytes_read=logical_bytes_read,
        parameter_read_seconds=read_seconds_total,
        materialization_seconds=materialization_seconds_total,
        compute_seconds=compute_seconds_total,
        release_seconds=release_seconds_total,
        maximum_workspace_bytes=maximum_workspace,
        output_activation_bytes=output_activation_bytes,
        output_checksum=checksum,
    )


def run_activation_recompute_validation(
    config: ExperimentConfig,
    *,
    parameter_store_path: str | Path,
    oracle_gradient_store_path: str | Path,
    gradient_store_path: str | Path,
    output_path: str | Path | None = None,
    device_override: str | None = None,
    parameter_working_set_bytes: int = 1024**2,
    gradient_working_set_bytes: int = 1024**2,
    activation_working_set_bytes: int = 1024**2,
    workspace_working_set_bytes: int = 4 * 1024**2,
) -> ActivationRecomputeResult:
    """Validate zero-boundary-retention recomputation against resident gradients."""

    if config.model.dropout != 0.0:
        raise ValueError("activation recomputation currently requires model.dropout=0")
    for value, name in (
        (parameter_working_set_bytes, "parameter_working_set_bytes"),
        (gradient_working_set_bytes, "gradient_working_set_bytes"),
        (activation_working_set_bytes, "activation_working_set_bytes"),
        (workspace_working_set_bytes, "workspace_working_set_bytes"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")

    parameter_destination = Path(parameter_store_path)
    oracle_destination = Path(oracle_gradient_store_path)
    gradient_destination = Path(gradient_store_path)
    for destination, label in (
        (parameter_destination, "parameter store"),
        (oracle_destination, "oracle gradient store"),
        (gradient_destination, "recomputed gradient store"),
    ):
        if destination.exists():
            raise FileExistsError(f"activation recomputation requires a new {label}: {destination}")

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
    transaction = parameter_store.begin_transaction(committed_step=0)
    transaction.put_many(bootstrap_payloads)
    parameter_commit = transaction.commit()
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
    oracle_store = VersionedTensorStore.create(
        oracle_destination,
        limits=_clone_limits(parameter_store),
    )
    oracle_transaction = oracle_store.begin_transaction(committed_step=0)
    oracle_transaction.put_many(resident.gradients)
    oracle_commit = oracle_transaction.commit()
    resident_loss = resident.loss
    resident_gradient_norm = resident.gradient_norm
    del resident, bootstrap_payloads
    gc.collect()

    recomputed_loss, forward_metrics, forward_workspace = (
        _forward_without_retained_boundaries(
            config,
            parameter_store,
            parameter_manifest.manifest_id,
            records,
            groups,
            input_ids,
            targets,
            device,
            parameter_working_set_bytes=parameter_working_set_bytes,
            workspace_working_set_bytes=workspace_working_set_bytes,
        )
    )
    gradient_store = VersionedTensorStore.create(
        gradient_destination,
        limits=_clone_limits(parameter_store),
    )

    upstream: Tensor | None = None
    backward_metrics: list[ActivationBackwardGroupMetrics] = []
    tied_accumulations = 0
    maximum_retained_activation = 0
    maximum_workspace = forward_workspace

    for reverse_ordinal, spec in enumerate(reversed(groups)):
        replay: PrefixReplayMetrics | None = None
        input_cpu: Tensor | None = None
        incoming_gradient_bytes = 0 if upstream is None else _activation_bytes(upstream)
        if spec.name != "embedding":
            input_cpu, replay = _recompute_group_input(
                config,
                parameter_store,
                parameter_manifest.manifest_id,
                records,
                groups,
                spec,
                input_ids,
                device,
                parameter_working_set_bytes=parameter_working_set_bytes,
                activation_working_set_bytes=activation_working_set_bytes,
                workspace_working_set_bytes=workspace_working_set_bytes,
            )
            simultaneous_cpu_bytes = replay.output_activation_bytes + incoming_gradient_bytes
            _check_activation(
                simultaneous_cpu_bytes,
                activation_working_set_bytes,
                f"recomputed input and incoming gradient for {spec.name}",
            )
            maximum_retained_activation = max(
                maximum_retained_activation,
                simultaneous_cpu_bytes,
            )
            maximum_workspace = max(maximum_workspace, replay.maximum_workspace_bytes)

        tensors, read_seconds, materialization_seconds, logical_bytes, chunk_reads = (
            _read_group(
                parameter_store,
                parameter_manifest.manifest_id,
                spec,
                records,
                device,
            )
        )
        _enable_parameter_gradients(tensors)
        input_activation_bytes = 0
        local_input: Tensor | None = None
        local_forward_started = time.perf_counter()

        if spec.name == "final-head":
            if input_cpu is None:
                raise RuntimeError("final head recomputation requires an input activation")
            local_input = input_cpu.to(device).detach()
            del input_cpu
            local_input.requires_grad_(True)
            input_activation_bytes = _activation_bytes(local_input)
            output = _final_forward(local_input, tensors, config)
            loss = F.cross_entropy(
                output.reshape(-1, output.size(-1)),
                targets.to(device).reshape(-1),
            )
            synchronize_accelerator(device)
            local_forward_seconds = time.perf_counter() - local_forward_started
            backward_started = time.perf_counter()
            loss.backward()
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            recomputed_loss = float(loss.detach().cpu().item())
            del loss
        elif spec.name.startswith("block-"):
            if input_cpu is None:
                raise RuntimeError(f"{spec.name} recomputation requires an input activation")
            if upstream is None:
                raise RuntimeError("block backward requires an upstream gradient")
            local_input = input_cpu.to(device).detach()
            del input_cpu
            local_input.requires_grad_(True)
            input_activation_bytes = _activation_bytes(local_input)
            output = _block_forward(
                local_input,
                tensors,
                config,
                int(spec.name.split("-", maxsplit=1)[1]),
            )
            synchronize_accelerator(device)
            local_forward_seconds = time.perf_counter() - local_forward_started
            upstream_device = upstream.to(device)
            backward_started = time.perf_counter()
            output.backward(upstream_device)
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            del upstream_device
            upstream = None
        else:
            if upstream is None:
                raise RuntimeError("embedding backward requires an upstream gradient")
            output = _embedding_forward(input_ids, tensors, device)
            synchronize_accelerator(device)
            local_forward_seconds = time.perf_counter() - local_forward_started
            upstream_device = upstream.to(device)
            backward_started = time.perf_counter()
            output.backward(upstream_device)
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            del upstream_device
            upstream = None

        output_activation_bytes = _activation_bytes(output)
        if local_input is not None:
            gradient_value = local_input.grad
            if gradient_value is None:
                raise RuntimeError(f"missing upstream gradient after {spec.name}")
            next_upstream_value = gradient_value.detach().cpu().contiguous()
            outgoing_gradient_bytes = _activation_bytes(next_upstream_value)
            next_upstream: Tensor | None = next_upstream_value
        else:
            next_upstream = None
            outgoing_gradient_bytes = 0

        _check_activation(
            outgoing_gradient_bytes,
            activation_working_set_bytes,
            f"outgoing activation gradient for {spec.name}",
        )
        maximum_retained_activation = max(
            maximum_retained_activation,
            outgoing_gradient_bytes,
        )
        local_workspace = (
            input_activation_bytes
            + output_activation_bytes
            + incoming_gradient_bytes
            + outgoing_gradient_bytes
        )
        _check_workspace(
            local_workspace,
            workspace_working_set_bytes,
            f"backward group {spec.name}",
        )
        maximum_workspace = max(maximum_workspace, local_workspace)

        extraction_started = time.perf_counter()
        gradients = _extract_group_gradients(tensors)
        gradient_payloads = tuple(
            _gradient_payload(name, gradient) for name, gradient in sorted(gradients.items())
        )
        extraction_seconds = time.perf_counter() - extraction_started
        local_gradient_checksum = _payload_digest(gradient_payloads)
        if "model.token_embedding.weight" in gradients:
            tied_accumulations += 1

        commit_started = time.perf_counter()
        commit_telemetry, committed_payloads = _commit_group_gradients(
            gradient_store,
            gradients,
        )
        commit_seconds = time.perf_counter() - commit_started
        del committed_payloads
        memory = accelerator_memory_metrics(device)
        process_rss = process_rss_bytes()

        release_started = time.perf_counter()
        del tensors, gradients, gradient_payloads, output
        if local_input is not None:
            del local_input
        gc.collect()
        synchronize_accelerator(device)
        release_seconds = time.perf_counter() - release_started
        release_memory = accelerator_memory_metrics(device)
        upstream = next_upstream

        backward_metrics.append(
            ActivationBackwardGroupMetrics(
                ordinal=reverse_ordinal,
                name=spec.name,
                tensor_names=spec.tensor_names,
                prefix_replay=replay,
                logical_parameter_bytes=logical_bytes,
                referenced_chunk_reads=chunk_reads,
                input_activation_bytes=input_activation_bytes,
                output_activation_bytes=output_activation_bytes,
                incoming_activation_gradient_bytes=incoming_gradient_bytes,
                outgoing_activation_gradient_bytes=outgoing_gradient_bytes,
                logical_workspace_bytes=local_workspace,
                parameter_read_seconds=read_seconds,
                materialization_seconds=materialization_seconds,
                local_forward_seconds=local_forward_seconds,
                backward_seconds=backward_seconds,
                gradient_extraction_seconds=extraction_seconds,
                gradient_commit_seconds=commit_seconds,
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

    oracle_norm = _stream_gradient_norm(oracle_store)
    recomputed_norm = _stream_gradient_norm(gradient_store)
    oracle_gradients = _read_all_gradients(oracle_store)
    recomputed_gradients = _read_all_gradients(gradient_store)
    comparison = compare_states(oracle_gradients, recomputed_gradients)
    gradient_manifest = gradient_store.current_manifest()
    tied_record = _gradient_map(gradient_manifest.tensors)["model.token_embedding.weight"]
    parameter_after = parameter_store.current_manifest()
    parameter_verification = parameter_store.verify()
    oracle_verification = oracle_store.verify()
    gradient_verification = gradient_store.verify()
    max_norm = config.training.gradient_clip_norm
    clip_coefficient = (
        1.0 if max_norm is None else min(1.0, max_norm / (recomputed_norm + 1e-6))
    )
    prefix_metrics = tuple(
        replay
        for replay in (item.prefix_replay for item in backward_metrics)
        if replay is not None
    )

    result = ActivationRecomputeResult(
        schema_version=ACTIVATION_RECOMPUTE_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        activation_policy="recompute",
        parameter_store_path=str(parameter_destination),
        oracle_gradient_store_path=str(oracle_destination),
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
        activation_working_set_budget_bytes=activation_working_set_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_bytes,
        maximum_parameter_group_bytes=maximum_parameter_group_bytes,
        maximum_gradient_group_bytes=maximum_gradient_group_bytes,
        maximum_retained_activation_bytes=maximum_retained_activation,
        maximum_workspace_bytes=maximum_workspace,
        parameter_budget_respected=maximum_parameter_group_bytes
        <= parameter_working_set_bytes,
        gradient_budget_respected=maximum_gradient_group_bytes
        <= gradient_working_set_bytes,
        activation_budget_respected=maximum_retained_activation
        <= activation_working_set_bytes,
        workspace_budget_respected=maximum_workspace <= workspace_working_set_bytes,
        retained_forward_boundary_count=0,
        retained_forward_boundary_bytes=0,
        forward_groups=forward_metrics,
        backward_groups=tuple(backward_metrics),
        backward_group_order=tuple(item.name for item in backward_metrics),
        total_prefix_replayed_groups=sum(
            len(item.replayed_group_names) for item in prefix_metrics
        ),
        total_prefix_parameter_tensor_reads=sum(
            item.parameter_tensor_reads for item in prefix_metrics
        ),
        total_prefix_parameter_chunk_reads=sum(
            item.parameter_chunk_reads for item in prefix_metrics
        ),
        total_prefix_parameter_logical_bytes_read=sum(
            item.parameter_logical_bytes_read for item in prefix_metrics
        ),
        total_prefix_recomputation_seconds=sum(
            item.compute_seconds for item in prefix_metrics
        ),
        resident_loss=resident_loss,
        recomputed_loss=recomputed_loss,
        loss_absolute_difference=abs(resident_loss - recomputed_loss),
        resident_gradient_norm=resident_gradient_norm,
        oracle_store_gradient_norm=oracle_norm,
        oracle_store_norm_absolute_difference=abs(resident_gradient_norm - oracle_norm),
        recomputed_gradient_norm=recomputed_norm,
        gradient_norm_absolute_difference=abs(resident_gradient_norm - recomputed_norm),
        future_clip_coefficient=clip_coefficient,
        gradient_tensor_count=len(gradient_manifest.tensors),
        tied_gradient_accumulation_count=tied_accumulations,
        tied_gradient_version=tied_record.version,
        resident_vs_recomputed_gradients=comparison,
        parameter_manifest_unchanged=(
            parameter_after.manifest_id == parameter_manifest.manifest_id
            and parameter_after.manifest_checksum == parameter_manifest.manifest_checksum
        ),
        parameter_store_verified_tensor_count=parameter_verification.tensor_count,
        oracle_gradient_store_verified_tensor_count=oracle_verification.tensor_count,
        recomputed_gradient_store_verified_tensor_count=gradient_verification.tensor_count,
    )
    if output_path is not None:
        write_json_atomic(output_path, result)
    del oracle_gradients, recomputed_gradients
    _release_accelerator(device)
    return result
