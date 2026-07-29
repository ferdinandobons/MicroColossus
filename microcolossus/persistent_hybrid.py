"""Nearest-anchor hybrid backward execution from an authoritative parameter store."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .activation_planner import (
    ACTIVATION_PLANNER_VERSION,
    ActivationMeasurementProfile,
    ActivationPlan,
    ActivationPlanIntegrityError,
)
from .activation_recompute import (
    ActivationBackwardGroupMetrics,
    ActivationForwardGroupMetrics,
    ActivationRecomputeResult,
    PrefixReplayMetrics,
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
from .storage.schema import TensorRecord
from .storage_training import compare_states
from .telemetry import (
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training_checkpoint import _parameter_payloads

ACTIVATION_HYBRID_SCHEMA_VERSION = "microcolossus.activation-hybrid.v1"


def _validate_hybrid_plan(
    profile: ActivationMeasurementProfile,
    plan: ActivationPlan,
    groups: tuple[ExecutionGroupSpec, ...],
    *,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> None:
    plan.validate(profile)
    if not plan.feasible:
        raise ActivationPlanIntegrityError(
            f"hybrid activation plan is infeasible: {plan.rejection_reason}"
        )
    if plan.selected_policy != "measured_budget_v1":
        raise ActivationPlanIntegrityError(
            f"unsupported selected activation policy: {plan.selected_policy}"
        )
    if plan.planner_version != ACTIVATION_PLANNER_VERSION:
        raise ActivationPlanIntegrityError(
            f"unsupported activation planner: {plan.planner_version}"
        )
    if plan.activation_budget_bytes != activation_working_set_bytes:
        raise ActivationPlanIntegrityError("activation budget does not match plan")
    if plan.workspace_budget_bytes != workspace_working_set_bytes:
        raise ActivationPlanIntegrityError("workspace budget does not match plan")
    group_names = tuple(item.name for item in groups)
    profile_names = tuple(item.name for item in profile.groups)
    if group_names != profile_names:
        raise ActivationPlanIntegrityError("profile group order does not match runtime")
    anchor_names = set(plan.selected_anchor_group_names)
    if "final-head" in anchor_names:
        raise ActivationPlanIntegrityError("final-head cannot be a hybrid anchor")
    for name in anchor_names:
        if name not in group_names:
            raise ActivationPlanIntegrityError(f"unknown hybrid anchor: {name}")
        if group_names.index(name) >= len(group_names) - 1:
            raise ActivationPlanIntegrityError(f"group cannot be a hybrid anchor: {name}")


def _segment_replay_names(
    groups: tuple[ExecutionGroupSpec, ...],
    *,
    target: ExecutionGroupSpec,
    anchor_group: str | None,
) -> tuple[str, ...]:
    if target.ordinal <= 0:
        return ()
    if anchor_group is None:
        start = 0
    else:
        anchor_ordinal = next(
            item.ordinal for item in groups if item.name == anchor_group
        )
        if anchor_ordinal >= target.ordinal:
            raise ActivationPlanIntegrityError(
                f"anchor {anchor_group} does not precede target {target.name}"
            )
        start = anchor_ordinal + 1
    return tuple(item.name for item in groups[start : target.ordinal])


def _forward_with_hybrid_anchors(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, TensorRecord],
    groups: tuple[ExecutionGroupSpec, ...],
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    *,
    anchor_names: set[str],
    parameter_working_set_bytes: int,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[
    float,
    tuple[ActivationForwardGroupMetrics, ...],
    dict[str, Tensor],
    int,
    int,
]:
    hidden_states: Tensor | None = None
    metrics: list[ActivationForwardGroupMetrics] = []
    anchors: dict[str, Tensor] = {}
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
                f"hybrid forward group {spec.name}",
            )
            maximum_workspace = max(maximum_workspace, workspace_bytes)
            output_cpu = output.detach().cpu().contiguous()
            output_checksum = tensor_checksum(output_cpu)
            retained_after_group = spec.name in anchor_names
            if retained_after_group:
                anchors[spec.name] = output_cpu
                retained_bytes = sum(_activation_bytes(value) for value in anchors.values())
                _check_activation(
                    retained_bytes,
                    activation_working_set_bytes,
                    "retained hybrid anchors during forward",
                )
            memory = accelerator_memory_metrics(device)
            process_rss = process_rss_bytes()

            release_started = time.perf_counter()
            del tensors
            if not retained_after_group:
                del output_cpu
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
                    retained_after_group=retained_after_group,
                    process_rss_after_compute_bytes=process_rss,
                    accelerator_after_compute=memory,
                )
            )

    if bounded_loss is None:
        raise RuntimeError("hybrid forward did not produce a loss")
    retained_anchor_bytes = sum(_activation_bytes(value) for value in anchors.values())
    _release_accelerator(device)
    return (
        bounded_loss,
        tuple(metrics),
        anchors,
        retained_anchor_bytes,
        maximum_workspace,
    )


def _hybrid_group_input(
    config: ExperimentConfig,
    store: VersionedTensorStore,
    manifest_id: str,
    records: dict[str, TensorRecord],
    groups: tuple[ExecutionGroupSpec, ...],
    target: ExecutionGroupSpec,
    segment_anchor: str | None,
    segment_replayed: tuple[str, ...],
    anchors: dict[str, Tensor],
    input_ids: Tensor,
    device: torch.device,
    *,
    parameter_working_set_bytes: int,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[Tensor, PrefixReplayMetrics]:
    if target.ordinal <= 0:
        raise ValueError("embedding does not consume a hidden activation")
    expected_replay = _segment_replay_names(
        groups,
        target=target,
        anchor_group=segment_anchor,
    )
    if expected_replay != segment_replayed:
        raise ActivationPlanIntegrityError(
            f"plan replay segment for {target.name} does not match runtime group order"
        )
    if segment_anchor is not None and not segment_replayed:
        anchor_cpu = anchors[segment_anchor].detach().cpu().contiguous()
        output_activation_bytes = _activation_bytes(anchor_cpu)
        _check_activation(
            output_activation_bytes,
            activation_working_set_bytes,
            f"hybrid anchor input for {target.name}",
        )
        return anchor_cpu, PrefixReplayMetrics(
            target_group=target.name,
            replayed_group_names=(),
            parameter_tensor_reads=0,
            parameter_chunk_reads=0,
            parameter_logical_bytes_read=0,
            parameter_read_seconds=0.0,
            materialization_seconds=0.0,
            compute_seconds=0.0,
            release_seconds=0.0,
            maximum_workspace_bytes=output_activation_bytes,
            output_activation_bytes=output_activation_bytes,
            output_checksum=tensor_checksum(anchor_cpu),
        )

    hidden_states: Tensor | None
    if segment_anchor is None:
        hidden_states = None
    else:
        if segment_anchor not in anchors:
            raise ActivationPlanIntegrityError(f"missing retained anchor: {segment_anchor}")
        hidden_states = anchors[segment_anchor].to(device)

    tensor_reads = 0
    chunk_reads = 0
    logical_bytes_read = 0
    read_seconds_total = 0.0
    materialization_seconds_total = 0.0
    compute_seconds_total = 0.0
    release_seconds_total = 0.0
    maximum_workspace = 0
    group_by_name = {item.name: item for item in groups}

    with torch.no_grad():
        for group_name in segment_replayed:
            spec = group_by_name[group_name]
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
                    f"hybrid replay group {spec.name} requires {logical_bytes} parameter bytes"
                )
            compute_started = time.perf_counter()
            if spec.name == "embedding":
                if hidden_states is not None:
                    raise RuntimeError("embedding replay cannot start from an anchor")
                output = _embedding_forward(input_ids, tensors, device)
            elif spec.name.startswith("block-"):
                if hidden_states is None:
                    raise RuntimeError("hybrid replay block executed before embeddings")
                output = _block_forward(
                    hidden_states,
                    tensors,
                    config,
                    int(spec.name.split("-", maxsplit=1)[1]),
                )
            else:
                raise RuntimeError("final head cannot be part of hybrid replay")
            synchronize_accelerator(device)
            compute_seconds = time.perf_counter() - compute_started
            output_activation_bytes = _activation_bytes(output)
            workspace_bytes = input_activation_bytes + output_activation_bytes
            _check_workspace(
                workspace_bytes,
                workspace_working_set_bytes,
                f"hybrid replay group {spec.name}",
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

            tensor_reads += len(spec.tensor_names)
            chunk_reads += chunks
            logical_bytes_read += logical_bytes
            read_seconds_total += read_seconds
            materialization_seconds_total += materialization_seconds
            compute_seconds_total += compute_seconds

    if hidden_states is None:
        raise RuntimeError(f"hybrid replay for {target.name} produced no activation")
    output_cpu = hidden_states.detach().cpu().contiguous()
    output_activation_bytes = _activation_bytes(output_cpu)
    _check_activation(
        output_activation_bytes,
        activation_working_set_bytes,
        f"hybrid replay input for {target.name}",
    )
    checksum = tensor_checksum(output_cpu)
    del hidden_states
    _release_accelerator(device)
    return output_cpu, PrefixReplayMetrics(
        target_group=target.name,
        replayed_group_names=segment_replayed,
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


def run_activation_hybrid_from_store(
    config: ExperimentConfig,
    *,
    parameter_store: VersionedTensorStore,
    activation_profile: ActivationMeasurementProfile,
    activation_plan: ActivationPlan,
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
) -> ActivationRecomputeResult:
    """Run one hybrid nearest-anchor backward pass from a parameter store."""

    if config.model.dropout != 0.0:
        raise ValueError("hybrid activation execution currently requires model.dropout=0")
    for value, name in (
        (parameter_working_set_bytes, "parameter_working_set_bytes"),
        (gradient_working_set_bytes, "gradient_working_set_bytes"),
        (activation_working_set_bytes, "activation_working_set_bytes"),
        (workspace_working_set_bytes, "workspace_working_set_bytes"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if oracle_gradient_store_path.exists():
        raise FileExistsError(f"oracle gradient store exists: {oracle_gradient_store_path}")
    if gradient_store_path.exists():
        raise FileExistsError(f"hybrid gradient store exists: {gradient_store_path}")

    parameter_manifest = parameter_store.current_manifest()
    records = _record_map(parameter_manifest.tensors)
    groups = build_execution_groups(config, set(records))
    _validate_hybrid_plan(
        activation_profile,
        activation_plan,
        groups,
        activation_working_set_bytes=activation_working_set_bytes,
        workspace_working_set_bytes=workspace_working_set_bytes,
    )
    segment_by_target = {item.target_group: item for item in activation_plan.replay_segments}
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

    probe_model = DecoderOnlyTransformer(config.model)
    parameter_count = probe_model.parameter_count
    del probe_model
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
    oracle_transaction = oracle_store.begin_transaction(committed_step=0)
    oracle_transaction.put_many(resident.gradients)
    oracle_commit = oracle_transaction.commit()
    resident_loss = resident.loss
    resident_gradient_norm = resident.gradient_norm
    del resident, resident_payloads
    gc.collect()

    hybrid_loss, forward_metrics, anchor_activations, retained_anchor_bytes, forward_max = (
        _forward_with_hybrid_anchors(
            config,
            parameter_store,
            parameter_manifest.manifest_id,
            records,
            groups,
            input_ids,
            targets,
            device,
            anchor_names=set(activation_plan.selected_anchor_group_names),
            parameter_working_set_bytes=parameter_working_set_bytes,
            activation_working_set_bytes=activation_working_set_bytes,
            workspace_working_set_bytes=workspace_working_set_bytes,
        )
    )
    gradient_store = VersionedTensorStore.create(
        gradient_store_path,
        limits=_clone_limits(parameter_store),
    )

    upstream: Tensor | None = None
    backward_metrics: list[ActivationBackwardGroupMetrics] = []
    tied_accumulations = 0
    maximum_retained_activation = retained_anchor_bytes
    maximum_workspace = forward_max

    for reverse_ordinal, spec in enumerate(reversed(groups)):
        replay: PrefixReplayMetrics | None = None
        input_cpu: Tensor | None = None
        incoming_gradient_bytes = 0 if upstream is None else _activation_bytes(upstream)
        if spec.name != "embedding":
            segment = segment_by_target.get(spec.name)
            if segment is None:
                raise ActivationPlanIntegrityError(f"missing replay segment for {spec.name}")
            input_cpu, replay = _hybrid_group_input(
                config,
                parameter_store,
                parameter_manifest.manifest_id,
                records,
                groups,
                spec,
                segment.anchor_group,
                segment.replayed_group_names,
                anchor_activations,
                input_ids,
                device,
                parameter_working_set_bytes=parameter_working_set_bytes,
                activation_working_set_bytes=activation_working_set_bytes,
                workspace_working_set_bytes=workspace_working_set_bytes,
            )
            retained_before_backward = retained_anchor_bytes + incoming_gradient_bytes
            _check_activation(
                retained_before_backward,
                activation_working_set_bytes,
                f"hybrid retained anchors and incoming gradient for {spec.name}",
            )
            maximum_retained_activation = max(
                maximum_retained_activation,
                retained_before_backward,
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
                raise RuntimeError("final head hybrid replay requires an input activation")
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
            hybrid_loss = float(loss.detach().cpu().item())
            del loss
        elif spec.name.startswith("block-"):
            if input_cpu is None:
                raise RuntimeError(f"{spec.name} hybrid replay requires an input activation")
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

        retained_after_backward = retained_anchor_bytes + outgoing_gradient_bytes
        _check_activation(
            retained_after_backward,
            activation_working_set_bytes,
            f"hybrid retained anchors and outgoing gradient for {spec.name}",
        )
        maximum_retained_activation = max(
            maximum_retained_activation,
            retained_after_backward,
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
            f"hybrid backward group {spec.name}",
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
    clip_coefficient = (
        1.0 if max_norm is None else min(1.0, max_norm / (hybrid_norm + 1e-6))
    )
    prefix_metrics = tuple(
        replay
        for replay in (item.prefix_replay for item in backward_metrics)
        if replay is not None
    )

    result = ActivationRecomputeResult(
        schema_version=ACTIVATION_HYBRID_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        activation_policy="hybrid",
        parameter_store_path=str(parameter_store.root),
        oracle_gradient_store_path=str(oracle_gradient_store_path),
        gradient_store_path=str(gradient_store_path),
        parameter_manifest_id=parameter_manifest.manifest_id,
        parameter_manifest_checksum=parameter_manifest.manifest_checksum,
        oracle_gradient_manifest_id=oracle_commit.manifest.manifest_id,
        gradient_manifest_id=gradient_manifest.manifest_id,
        oracle_gradient_store_commit=oracle_commit.telemetry,
        parameter_count=parameter_count,
        batch_checksum=_batch_checksum(input_ids, targets),
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
        retained_forward_boundary_count=len(anchor_activations),
        retained_forward_boundary_bytes=retained_anchor_bytes,
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
        recomputed_loss=hybrid_loss,
        loss_absolute_difference=abs(resident_loss - hybrid_loss),
        resident_gradient_norm=resident_gradient_norm,
        oracle_store_gradient_norm=oracle_norm,
        oracle_store_norm_absolute_difference=abs(resident_gradient_norm - oracle_norm),
        recomputed_gradient_norm=hybrid_norm,
        gradient_norm_absolute_difference=abs(resident_gradient_norm - hybrid_norm),
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
        activation_profile_checksum=activation_profile.profile_checksum,
        activation_plan_checksum=activation_plan.plan_checksum,
        activation_planner_version=activation_plan.planner_version,
        selected_anchor_group_names=activation_plan.selected_anchor_group_names,
    )
    write_json_atomic(output_path, result)
    del oracle_gradients, hybrid_gradients, anchor_activations
    _release_accelerator(device)
    return result
