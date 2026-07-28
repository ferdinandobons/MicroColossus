"""Persistent hybrid-anchor backward execution from an authoritative parameter store."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .activation_planner import ActivationPlan, validate_plan_for_config
from .activation_recompute import (
    ActivationWorkingSetExceededError,
    WorkspaceWorkingSetExceededError,
    _check_activation,
    _check_workspace,
    _clone_limits,
)
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
    build_execution_groups,
    tensor_checksum,
)
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .storage import VersionedTensorStore
from .storage_training import StateComparison, compare_states
from .telemetry import (
    AcceleratorMemoryMetrics,
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training_checkpoint import _parameter_payloads

HYBRID_ACTIVATION_SCHEMA_VERSION = "microcolossus.hybrid-activation.v1"


@dataclass(frozen=True)
class HybridForwardGroupMetrics:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    logical_parameter_bytes: int
    referenced_chunk_reads: int
    input_activation_bytes: int
    output_activation_bytes: int
    logical_workspace_bytes: int
    retained_as_anchor: bool
    retained_anchor_bytes_after_group: int
    output_checksum: str
    parameter_read_seconds: float
    materialization_seconds: float
    compute_seconds: float
    release_seconds: float
    process_rss_after_compute_bytes: int
    accelerator_after_compute: AcceleratorMemoryMetrics


@dataclass(frozen=True)
class HybridReplayMetrics:
    target_group: str
    source_anchor_group: str | None
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
class HybridBackwardGroupMetrics:
    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    replay: HybridReplayMetrics | None
    logical_parameter_bytes: int
    referenced_chunk_reads: int
    input_activation_bytes: int
    output_activation_bytes: int
    incoming_activation_gradient_bytes: int
    outgoing_activation_gradient_bytes: int
    retained_anchor_bytes_before_group: int
    retained_anchor_bytes_after_group: int
    logical_activation_residency_bytes: int
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
class HybridActivationResult:
    schema_version: str
    experiment: str
    device: str
    activation_policy: str
    activation_plan_checksum: str
    activation_profile_checksum: str
    anchor_group_names: tuple[str, ...]
    parameter_store_path: str
    oracle_gradient_store_path: str
    gradient_store_path: str
    parameter_manifest_id: str
    parameter_manifest_checksum: str
    oracle_gradient_manifest_id: str
    gradient_manifest_id: str
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
    forward_groups: tuple[HybridForwardGroupMetrics, ...]
    backward_groups: tuple[HybridBackwardGroupMetrics, ...]
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
    all_anchors_released: bool
    full_gradient_state_materialized_for_validation: bool = True
    resident_oracle_materialized_for_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _anchor_bytes(anchors: dict[str, Tensor]) -> int:
    return sum(_activation_bytes(value) for value in anchors.values())


def _group_map(groups: tuple[ExecutionGroupSpec, ...]) -> dict[str, ExecutionGroupSpec]:
    return {item.name: item for item in groups}


def _hybrid_forward(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, Any],
    groups: tuple[ExecutionGroupSpec, ...],
    plan: ActivationPlan,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    *,
    parameter_working_set_bytes: int,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[
    float,
    tuple[HybridForwardGroupMetrics, ...],
    dict[str, Tensor],
    int,
    int,
]:
    anchors: dict[str, Tensor] = {}
    anchor_names = set(plan.anchor_group_names)
    hidden_states: Tensor | None = None
    metrics: list[HybridForwardGroupMetrics] = []
    maximum_workspace = 0
    maximum_activation = 0
    loss_value: float | None = None

    with torch.no_grad():
        for spec in groups:
            input_bytes = (
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
            started = time.perf_counter()
            if spec.name == "embedding":
                output = _embedding_forward(input_ids, tensors, device)
            elif spec.name.startswith("block-"):
                if hidden_states is None:
                    raise RuntimeError("hybrid block executed before embeddings")
                output = _block_forward(
                    hidden_states,
                    tensors,
                    config,
                    int(spec.name.split("-", maxsplit=1)[1]),
                )
            else:
                if hidden_states is None:
                    raise RuntimeError("hybrid final head executed before hidden states")
                output = _final_forward(hidden_states, tensors, config)
                loss = F.cross_entropy(
                    output.reshape(-1, output.size(-1)),
                    targets.to(device).reshape(-1),
                )
                loss_value = float(loss.detach().cpu().item())
                del loss
            synchronize_accelerator(device)
            compute_seconds = time.perf_counter() - started
            output_bytes = _activation_bytes(output)
            workspace = input_bytes + output_bytes
            _check_workspace(
                workspace,
                workspace_working_set_bytes,
                f"hybrid forward group {spec.name}",
            )
            maximum_workspace = max(maximum_workspace, workspace)
            output_cpu = output.detach().cpu().contiguous()
            checksum = tensor_checksum(output_cpu)
            if spec.name in anchor_names:
                anchors[spec.name] = output_cpu
            retained = _anchor_bytes(anchors)
            _check_activation(
                retained,
                activation_working_set_bytes,
                f"hybrid anchors after {spec.name}",
            )
            maximum_activation = max(maximum_activation, retained)
            memory = accelerator_memory_metrics(device)
            rss = process_rss_bytes()
            release_started = time.perf_counter()
            if spec.name not in anchor_names:
                del output_cpu
            del tensors
            if spec.name == "final-head":
                previous = hidden_states
                hidden_states = None
                if previous is not None:
                    del previous
                del output
            else:
                previous = hidden_states
                hidden_states = output
                if previous is not None:
                    del previous
            gc.collect()
            synchronize_accelerator(device)
            release_seconds = time.perf_counter() - release_started
            metrics.append(
                HybridForwardGroupMetrics(
                    ordinal=spec.ordinal,
                    name=spec.name,
                    tensor_names=spec.tensor_names,
                    logical_parameter_bytes=logical_bytes,
                    referenced_chunk_reads=chunks,
                    input_activation_bytes=input_bytes,
                    output_activation_bytes=output_bytes,
                    logical_workspace_bytes=workspace,
                    retained_as_anchor=spec.name in anchor_names,
                    retained_anchor_bytes_after_group=retained,
                    output_checksum=checksum,
                    parameter_read_seconds=read_seconds,
                    materialization_seconds=materialization_seconds,
                    compute_seconds=compute_seconds,
                    release_seconds=release_seconds,
                    process_rss_after_compute_bytes=rss,
                    accelerator_after_compute=memory,
                )
            )
    if loss_value is None:
        raise RuntimeError("hybrid forward did not produce a loss")
    _release_accelerator(device)
    return loss_value, tuple(metrics), anchors, maximum_activation, maximum_workspace


def _replay_input(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, Any],
    groups_by_name: dict[str, ExecutionGroupSpec],
    segment: Any,
    anchors: dict[str, Tensor],
    input_ids: Tensor,
    device: torch.device,
    *,
    parameter_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[Tensor, HybridReplayMetrics]:
    source = segment.source_anchor_group
    hidden_states: Tensor | None = None
    if source is not None:
        try:
            hidden_states = anchors[source].to(device)
        except KeyError as exc:
            raise RuntimeError(f"missing retained activation anchor: {source}") from exc
    tensor_reads = 0
    chunk_reads = 0
    logical_bytes = 0
    read_seconds = 0.0
    materialization_seconds = 0.0
    compute_seconds = 0.0
    release_seconds = 0.0
    maximum_workspace = 0

    with torch.no_grad():
        for name in segment.replayed_group_names:
            spec = groups_by_name[name]
            input_bytes = (
                _activation_bytes(input_ids)
                if hidden_states is None
                else _activation_bytes(hidden_states)
            )
            tensors, read_time, materialization_time, group_bytes, chunks = _read_group(
                store,
                manifest_id,
                spec,
                records,
                device,
            )
            if group_bytes > parameter_working_set_bytes:
                raise WorkingSetExceededError(
                    f"hybrid replay group {name} requires {group_bytes} parameter bytes"
                )
            started = time.perf_counter()
            if name == "embedding":
                output = _embedding_forward(input_ids, tensors, device)
            elif name.startswith("block-"):
                if hidden_states is None:
                    raise RuntimeError("hybrid replay block executed before embeddings")
                output = _block_forward(
                    hidden_states,
                    tensors,
                    config,
                    int(name.split("-", maxsplit=1)[1]),
                )
            else:
                raise RuntimeError("final-head cannot be replayed as a prefix")
            synchronize_accelerator(device)
            group_compute = time.perf_counter() - started
            output_bytes = _activation_bytes(output)
            workspace = input_bytes + output_bytes
            _check_workspace(
                workspace,
                workspace_working_set_bytes,
                f"hybrid replay group {name}",
            )
            maximum_workspace = max(maximum_workspace, workspace)
            release_started = time.perf_counter()
            del tensors
            previous = hidden_states
            hidden_states = output
            if previous is not None:
                del previous
            gc.collect()
            synchronize_accelerator(device)
            release_seconds += time.perf_counter() - release_started
            tensor_reads += len(spec.tensor_names)
            chunk_reads += chunks
            logical_bytes += group_bytes
            read_seconds += read_time
            materialization_seconds += materialization_time
            compute_seconds += group_compute

    if hidden_states is None:
        raise RuntimeError(f"hybrid replay for {segment.target_group} produced no activation")
    output_cpu = hidden_states.detach().cpu().contiguous()
    output_bytes = _activation_bytes(output_cpu)
    checksum = tensor_checksum(output_cpu)
    del hidden_states
    _release_accelerator(device)
    return output_cpu, HybridReplayMetrics(
        target_group=segment.target_group,
        source_anchor_group=source,
        replayed_group_names=tuple(segment.replayed_group_names),
        parameter_tensor_reads=tensor_reads,
        parameter_chunk_reads=chunk_reads,
        parameter_logical_bytes_read=logical_bytes,
        parameter_read_seconds=read_seconds,
        materialization_seconds=materialization_seconds,
        compute_seconds=compute_seconds,
        release_seconds=release_seconds,
        maximum_workspace_bytes=maximum_workspace,
        output_activation_bytes=output_bytes,
        output_checksum=checksum,
    )


def run_hybrid_activation_from_store(
    config: ExperimentConfig,
    *,
    activation_plan: ActivationPlan,
    parameter_store: VersionedTensorStore,
    oracle_gradient_store_path: Path,
    gradient_store_path: Path,
    output_path: Path,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    parameter_working_set_bytes: int,
    gradient_working_set_bytes: int,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> HybridActivationResult:
    """Run one hybrid-anchor backward step and compare it with resident gradients."""

    if config.model.dropout != 0.0:
        raise ValueError("hybrid activation execution currently requires model.dropout=0")
    validate_plan_for_config(
        config,
        activation_plan,
        activation_working_set_budget_bytes=activation_working_set_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_bytes,
    )
    for destination, label in (
        (oracle_gradient_store_path, "oracle gradient store"),
        (gradient_store_path, "hybrid gradient store"),
    ):
        if destination.exists():
            raise FileExistsError(f"{label} exists: {destination}")

    parameter_manifest = parameter_store.current_manifest()
    records = _record_map(parameter_manifest.tensors)
    groups = build_execution_groups(config, set(records))
    groups_by_name = _group_map(groups)
    if tuple(item.name for item in groups) != tuple(
        ["embedding", *[f"block-{i}" for i in range(config.model.layers)], "final-head"]
    ):
        raise RuntimeError("unexpected execution-group order")
    segment_by_target = {item.target_group: item for item in activation_plan.segments}
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

    probe = DecoderOnlyTransformer(config.model)
    parameter_count = probe.parameter_count
    del probe
    resident_payloads = _parameter_payloads(parameter_store)
    resident = _resident_gradient_trace(
        config,
        resident_payloads,
        input_ids,
        targets,
        device,
    )
    oracle_store = VersionedTensorStore.create(
        oracle_gradient_store_path,
        limits=_clone_limits(parameter_store),
    )
    transaction = oracle_store.begin_transaction(committed_step=0)
    transaction.put_many(resident.gradients)
    oracle_commit = transaction.commit()
    resident_loss = resident.loss
    resident_gradient_norm = resident.gradient_norm
    del resident, resident_payloads
    gc.collect()

    (
        recomputed_loss,
        forward_metrics,
        anchors,
        maximum_activation,
        maximum_workspace,
    ) = _hybrid_forward(
        config,
        parameter_store,
        parameter_manifest.manifest_id,
        records,
        groups,
        activation_plan,
        input_ids,
        targets,
        device,
        parameter_working_set_bytes=parameter_working_set_bytes,
        activation_working_set_bytes=activation_working_set_bytes,
        workspace_working_set_bytes=workspace_working_set_bytes,
    )
    gradient_store = VersionedTensorStore.create(
        gradient_store_path,
        limits=_clone_limits(parameter_store),
    )

    upstream: Tensor | None = None
    backward_metrics: list[HybridBackwardGroupMetrics] = []
    tied_accumulations = 0
    for reverse_ordinal, spec in enumerate(reversed(groups)):
        replay: HybridReplayMetrics | None = None
        input_cpu: Tensor | None = None
        incoming_bytes = 0 if upstream is None else _activation_bytes(upstream)
        retained_before = _anchor_bytes(anchors)
        if spec.name != "embedding":
            direct_source = groups[spec.ordinal - 1].name
            if direct_source in anchors:
                input_cpu = anchors.pop(direct_source)
                replay = HybridReplayMetrics(
                    target_group=spec.name,
                    source_anchor_group=direct_source,
                    replayed_group_names=(),
                    parameter_tensor_reads=0,
                    parameter_chunk_reads=0,
                    parameter_logical_bytes_read=0,
                    parameter_read_seconds=0.0,
                    materialization_seconds=0.0,
                    compute_seconds=0.0,
                    release_seconds=0.0,
                    maximum_workspace_bytes=0,
                    output_activation_bytes=_activation_bytes(input_cpu),
                    output_checksum=tensor_checksum(input_cpu),
                )
            else:
                try:
                    segment = segment_by_target[spec.name]
                except KeyError as exc:
                    raise RuntimeError(f"missing hybrid segment for {spec.name}") from exc
                input_cpu, replay = _replay_input(
                    config,
                    parameter_store,
                    parameter_manifest.manifest_id,
                    records,
                    groups_by_name,
                    segment,
                    anchors,
                    input_ids,
                    device,
                    parameter_working_set_bytes=parameter_working_set_bytes,
                    workspace_working_set_bytes=workspace_working_set_bytes,
                )
                maximum_workspace = max(maximum_workspace, replay.maximum_workspace_bytes)
            activation_residency = (
                _anchor_bytes(anchors) + _activation_bytes(input_cpu) + incoming_bytes
            )
            _check_activation(
                activation_residency,
                activation_working_set_bytes,
                f"hybrid input, anchors, and incoming gradient for {spec.name}",
            )
            maximum_activation = max(maximum_activation, activation_residency)

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
        local_started = time.perf_counter()
        if spec.name == "final-head":
            if input_cpu is None:
                raise RuntimeError("hybrid final head requires an input activation")
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
            local_forward_seconds = time.perf_counter() - local_started
            backward_started = time.perf_counter()
            loss.backward()
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            recomputed_loss = float(loss.detach().cpu().item())
            del loss
        elif spec.name.startswith("block-"):
            if input_cpu is None or upstream is None:
                raise RuntimeError(f"hybrid {spec.name} requires input and upstream gradient")
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
            local_forward_seconds = time.perf_counter() - local_started
            upstream_device = upstream.to(device)
            backward_started = time.perf_counter()
            output.backward(upstream_device)
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            del upstream_device
            upstream = None
        else:
            if upstream is None:
                raise RuntimeError("hybrid embedding backward requires upstream gradient")
            output = _embedding_forward(input_ids, tensors, device)
            synchronize_accelerator(device)
            local_forward_seconds = time.perf_counter() - local_started
            upstream_device = upstream.to(device)
            backward_started = time.perf_counter()
            output.backward(upstream_device)
            synchronize_accelerator(device)
            backward_seconds = time.perf_counter() - backward_started
            del upstream_device
            upstream = None

        output_activation_bytes = _activation_bytes(output)
        if local_input is not None:
            if local_input.grad is None:
                raise RuntimeError(f"missing upstream gradient after {spec.name}")
            next_upstream = local_input.grad.detach().cpu().contiguous()
            outgoing_bytes = _activation_bytes(next_upstream)
        else:
            next_upstream = None
            outgoing_bytes = 0
        retained_after = _anchor_bytes(anchors)
        activation_after = retained_after + outgoing_bytes
        _check_activation(
            activation_after,
            activation_working_set_bytes,
            f"hybrid anchors and outgoing gradient after {spec.name}",
        )
        maximum_activation = max(maximum_activation, activation_after)
        workspace = (
            input_activation_bytes
            + output_activation_bytes
            + incoming_bytes
            + outgoing_bytes
        )
        _check_workspace(
            workspace,
            workspace_working_set_bytes,
            f"hybrid backward group {spec.name}",
        )
        maximum_workspace = max(maximum_workspace, workspace)

        extraction_started = time.perf_counter()
        gradients = _extract_group_gradients(tensors)
        payloads = tuple(
            _gradient_payload(name, gradient) for name, gradient in sorted(gradients.items())
        )
        extraction_seconds = time.perf_counter() - extraction_started
        local_checksum = _payload_digest(payloads)
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
        rss = process_rss_bytes()
        release_started = time.perf_counter()
        del tensors, gradients, payloads, output
        if local_input is not None:
            del local_input
        gc.collect()
        synchronize_accelerator(device)
        release_seconds = time.perf_counter() - release_started
        release_memory = accelerator_memory_metrics(device)
        upstream = next_upstream
        backward_metrics.append(
            HybridBackwardGroupMetrics(
                ordinal=reverse_ordinal,
                name=spec.name,
                tensor_names=spec.tensor_names,
                replay=replay,
                logical_parameter_bytes=logical_bytes,
                referenced_chunk_reads=chunk_reads,
                input_activation_bytes=input_activation_bytes,
                output_activation_bytes=output_activation_bytes,
                incoming_activation_gradient_bytes=incoming_bytes,
                outgoing_activation_gradient_bytes=outgoing_bytes,
                retained_anchor_bytes_before_group=retained_before,
                retained_anchor_bytes_after_group=retained_after,
                logical_activation_residency_bytes=max(
                    activation_after,
                    retained_after + input_activation_bytes + incoming_bytes,
                ),
                logical_workspace_bytes=workspace,
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
                local_gradient_checksum=local_checksum,
                upstream_gradient_checksum=(
                    None if next_upstream is None else tensor_checksum(next_upstream)
                ),
                process_rss_after_backward_bytes=rss,
                accelerator_after_backward=memory,
                accelerator_after_release=release_memory,
            )
        )

    oracle_norm = _stream_gradient_norm(oracle_store)
    hybrid_norm = _stream_gradient_norm(gradient_store)
    oracle_gradients = _read_all_gradients(oracle_store)
    hybrid_gradients = _read_all_gradients(gradient_store)
    comparison = compare_states(oracle_gradients, hybrid_gradients)
    gradient_manifest = gradient_store.current_manifest()
    tied_record = _gradient_map(gradient_manifest.tensors)["model.token_embedding.weight"]
    parameter_after = parameter_store.current_manifest()
    parameter_verification = parameter_store.verify()
    oracle_verification = oracle_store.verify()
    gradient_verification = gradient_store.verify()
    max_norm = config.training.gradient_clip_norm
    coefficient = 1.0 if max_norm is None else min(1.0, max_norm / (hybrid_norm + 1e-6))
    replay_metrics = tuple(
        item.replay for item in backward_metrics if item.replay is not None
    )
    result = HybridActivationResult(
        schema_version=HYBRID_ACTIVATION_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        activation_policy="hybrid",
        activation_plan_checksum=activation_plan.plan_checksum,
        activation_profile_checksum=activation_plan.profile_checksum,
        anchor_group_names=activation_plan.anchor_group_names,
        parameter_store_path=str(parameter_store.root),
        oracle_gradient_store_path=str(oracle_gradient_store_path),
        gradient_store_path=str(gradient_store_path),
        parameter_manifest_id=parameter_manifest.manifest_id,
        parameter_manifest_checksum=parameter_manifest.manifest_checksum,
        oracle_gradient_manifest_id=oracle_commit.manifest.manifest_id,
        gradient_manifest_id=gradient_manifest.manifest_id,
        parameter_count=parameter_count,
        batch_checksum=_batch_checksum(input_ids, targets),
        parameter_working_set_budget_bytes=parameter_working_set_bytes,
        gradient_working_set_budget_bytes=gradient_working_set_bytes,
        activation_working_set_budget_bytes=activation_working_set_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_bytes,
        maximum_parameter_group_bytes=maximum_parameter_group_bytes,
        maximum_gradient_group_bytes=maximum_gradient_group_bytes,
        maximum_retained_activation_bytes=maximum_activation,
        maximum_workspace_bytes=maximum_workspace,
        parameter_budget_respected=maximum_parameter_group_bytes
        <= parameter_working_set_bytes,
        gradient_budget_respected=maximum_gradient_group_bytes
        <= gradient_working_set_bytes,
        activation_budget_respected=maximum_activation <= activation_working_set_bytes,
        workspace_budget_respected=maximum_workspace <= workspace_working_set_bytes,
        retained_forward_boundary_count=len(activation_plan.anchor_group_names),
        retained_forward_boundary_bytes=activation_plan.maximum_retained_anchor_bytes,
        forward_groups=forward_metrics,
        backward_groups=tuple(backward_metrics),
        backward_group_order=tuple(item.name for item in backward_metrics),
        total_prefix_replayed_groups=sum(
            len(item.replayed_group_names) for item in replay_metrics
        ),
        total_prefix_parameter_tensor_reads=sum(
            item.parameter_tensor_reads for item in replay_metrics
        ),
        total_prefix_parameter_chunk_reads=sum(
            item.parameter_chunk_reads for item in replay_metrics
        ),
        total_prefix_parameter_logical_bytes_read=sum(
            item.parameter_logical_bytes_read for item in replay_metrics
        ),
        total_prefix_recomputation_seconds=math.fsum(
            item.compute_seconds for item in replay_metrics
        ),
        resident_loss=resident_loss,
        recomputed_loss=recomputed_loss,
        loss_absolute_difference=abs(resident_loss - recomputed_loss),
        resident_gradient_norm=resident_gradient_norm,
        oracle_store_gradient_norm=oracle_norm,
        oracle_store_norm_absolute_difference=abs(resident_gradient_norm - oracle_norm),
        recomputed_gradient_norm=hybrid_norm,
        gradient_norm_absolute_difference=abs(resident_gradient_norm - hybrid_norm),
        future_clip_coefficient=coefficient,
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
        all_anchors_released=not anchors,
    )
    write_json_atomic(output_path, result)
    del oracle_gradients, hybrid_gradients
    _release_accelerator(device)
    return result
