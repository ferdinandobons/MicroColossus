"""Activation-recomputed backward execution from an authoritative parameter store."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .activation_recompute import (
    ActivationBackwardGroupMetrics,
    ActivationRecomputeResult,
    PrefixReplayMetrics,
    _check_activation,
    _check_workspace,
    _clone_limits,
    _forward_without_retained_boundaries,
    _recompute_group_input,
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
from .storage_training import compare_states
from .telemetry import (
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training_checkpoint import _parameter_payloads


def run_activation_recompute_from_store(
    config: ExperimentConfig,
    *,
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
) -> ActivationRecomputeResult:
    """Recompute group inputs from an existing authoritative parameter store."""

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
    if oracle_gradient_store_path.exists():
        raise FileExistsError(f"oracle gradient store exists: {oracle_gradient_store_path}")
    if gradient_store_path.exists():
        raise FileExistsError(f"recomputed gradient store exists: {gradient_store_path}")

    parameter_manifest = parameter_store.current_manifest()
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
        gradient_store_path,
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
        schema_version="microcolossus.activation-recompute.v1",
        experiment=config.name,
        device=str(device),
        activation_policy="recompute",
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
    write_json_atomic(output_path, result)
    del oracle_gradients, recomputed_gradients
    _release_accelerator(device)
    return result
