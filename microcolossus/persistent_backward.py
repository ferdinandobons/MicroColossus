"""Bounded backward execution from an existing parameter store."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .bounded_backward import (
    BackwardGroupMetrics,
    BoundedBackwardResult,
    GradientWorkingSetExceededError,
    _batch_checksum,
    _bounded_forward_activations,
    _commit_group_gradients,
    _enable_parameter_gradients,
    _extract_group_gradients,
    _gradient_map,
    _gradient_payload,
    _group_input_activation,
    _read_all_gradients,
    _resident_gradient_trace,
    _stream_gradient_norm,
)
from .bounded_backward import _payload_digest as _gradient_payload_digest
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
from .storage import StoreLimits, VersionedTensorStore
from .storage_training import compare_states
from .telemetry import (
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training_checkpoint import _parameter_payloads


def run_bounded_backward_from_store(
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
) -> BoundedBackwardResult:
    if config.model.dropout != 0.0:
        raise ValueError("bounded training currently requires model.dropout=0")
    if parameter_working_set_bytes <= 0:
        raise ValueError("parameter_working_set_bytes must be greater than zero")
    if gradient_working_set_bytes <= 0:
        raise ValueError("gradient_working_set_bytes must be greater than zero")
    if oracle_gradient_store_path.exists():
        raise FileExistsError(f"oracle gradient store exists: {oracle_gradient_store_path}")
    if gradient_store_path.exists():
        raise FileExistsError(f"bounded gradient store exists: {gradient_store_path}")

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
    oracle_gradient_store = VersionedTensorStore.create(
        oracle_gradient_store_path,
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
    del resident, resident_payloads
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
        gradient_store_path,
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
        local_gradient_checksum = _gradient_payload_digest(gradient_payloads)
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
    data_checksum = _batch_checksum(input_ids, targets)
    result = BoundedBackwardResult(
        schema_version="microcolossus.bounded-backward.v1",
        experiment=config.name,
        device=str(device),
        parameter_store_path=str(parameter_store.root),
        oracle_gradient_store_path=str(oracle_gradient_store_path),
        gradient_store_path=str(gradient_store_path),
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
        oracle_gradient_store_verified_tensor_count=oracle_gradient_verification.tensor_count,
        oracle_gradient_store_verified_chunk_count=oracle_gradient_verification.chunk_count,
        gradient_store_verified_tensor_count=gradient_verification.tensor_count,
        gradient_store_verified_chunk_count=gradient_verification.chunk_count,
        gradient_versions=tuple(
            sorted((record.tensor_id, record.version) for record in gradient_manifest.tensors)
        ),
    )
    write_json_atomic(output_path, result)
    del activations, oracle_gradients, bounded_gradients
    _release_accelerator(device)
    return result
