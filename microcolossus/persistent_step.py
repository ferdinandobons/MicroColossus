"""Advance one persistent bounded training step from an authoritative bundle."""

from __future__ import annotations

import gc
import time
import uuid
from pathlib import Path

import torch
from torch import Tensor

from .activation_recompute import (
    ActivationWorkingSetExceededError,
    WorkspaceWorkingSetExceededError,
)
from .bounded_backward import _batch_checksum
from .bounded_forward import build_execution_groups
from .bounded_optimizer import (
    OptimizerGroupMetrics,
    OptimizerWorkingSetExceededError,
    _apply_adamw_group,
    _bare_parameter_name,
    _gradient_records,
    _optimizer_groups,
    _optimizer_records,
    _parameter_records,
    _payload_digest,
    _read_payloads,
    _store_limits,
    _store_payloads,
)
from .bounded_training_types import PersistentStepResult
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .persistent_backward import run_bounded_backward_from_store
from .persistent_recompute import run_activation_recompute_from_store
from .step_bundle import (
    BundleFailureInjector,
    StepBundleManifest,
    StepBundleStore,
)
from .storage import IntegrityError, TensorPayload, VersionedTensorStore
from .storage.adapters import export_pytorch_state, restore_pytorch_state
from .storage.schema import StoreTelemetry
from .storage_training import compare_states
from .telemetry import (
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
)
from .training_checkpoint import _batch_for_cursor, _open_referenced_store


def _resident_oracle_from_current_state(
    config: ExperimentConfig,
    *,
    parameter_store: VersionedTensorStore,
    optimizer_store: VersionedTensorStore,
    oracle_store_path: Path,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    clipping_coefficient: float,
    committed_step: int,
) -> tuple[float, StoreTelemetry]:
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=False,
    )
    state_payloads = tuple(
        sorted(
            _store_payloads(parameter_store) + _store_payloads(optimizer_store),
            key=lambda item: item.logical_name,
        )
    )
    restore_pytorch_state(model, state_payloads, optimizer=optimizer)
    del state_payloads
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids.to(device), targets.to(device))
    if output.loss is None:
        raise RuntimeError("resident multi-step oracle did not return a loss")
    output.loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(clipping_coefficient)
    synchronize_accelerator(device)
    optimizer.step()
    synchronize_accelerator(device)
    loss = float(output.loss.detach().cpu().item())
    oracle_payloads = export_pytorch_state(model, optimizer)
    oracle_store = VersionedTensorStore.create(
        oracle_store_path,
        limits=_store_limits(config),
    )
    transaction = oracle_store.begin_transaction(committed_step=committed_step)
    transaction.put_many(oracle_payloads)
    commit = transaction.commit()
    del oracle_payloads, output, optimizer, model
    gc.collect()
    synchronize_accelerator(device)
    return loss, commit.telemetry


def _commit_versioned_payloads(
    store: VersionedTensorStore,
    payloads: tuple[TensorPayload, ...],
    *,
    committed_step: int,
    version: int,
) -> StoreTelemetry:
    transaction = store.begin_transaction(committed_step=committed_step)
    for payload in payloads:
        transaction.put_tensor(payload, version=version)
    return transaction.commit().telemetry


def _retain_all_workspace_bytes(backward_groups: tuple[object, ...]) -> int:
    maximum = 0
    for item in backward_groups:
        required = (
            int(getattr(item, "input_activation_bytes"))
            + int(getattr(item, "output_activation_bytes"))
            + int(getattr(item, "incoming_activation_gradient_bytes"))
            + int(getattr(item, "outgoing_activation_gradient_bytes"))
        )
        maximum = max(maximum, required)
    return maximum


def advance_one_step(
    config: ExperimentConfig,
    *,
    bundle_store: StepBundleStore,
    current: StepBundleManifest,
    batch_cursor: int,
    device: torch.device,
    parameter_working_set_bytes: int,
    gradient_working_set_bytes: int,
    optimizer_working_set_bytes: int,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
    bundle_failure_injector: BundleFailureInjector | None,
) -> PersistentStepResult:
    next_step = current.committed_step + 1
    if batch_cursor != current.committed_step:
        raise IntegrityError("batch cursor and committed step must advance together")
    input_ids, targets, batch_seed = _batch_for_cursor(config, batch_cursor)
    data_checksum = _batch_checksum(input_ids, targets)
    attempt = f"step-{next_step}-{uuid.uuid4().hex}"
    work = bundle_store.root / "work" / attempt
    candidates = bundle_store.root / "candidates" / attempt
    backward_result_path = work / f"{config.training.activation_policy}-backward.json"
    oracle_gradient_store_path = work / "oracle-gradients"
    gradient_store_path = work / "bounded-gradients"
    oracle_state_store_path = work / "oracle-state"
    candidate_parameter_store_path = candidates / "parameters"
    candidate_optimizer_store_path = candidates / "optimizer"

    parameter_store = _open_referenced_store(bundle_store, current, kind="parameter")
    optimizer_store = _open_referenced_store(bundle_store, current, kind="optimizer")

    if config.training.activation_policy == "recompute":
        recomputed = run_activation_recompute_from_store(
            config,
            parameter_store=parameter_store,
            oracle_gradient_store_path=oracle_gradient_store_path,
            gradient_store_path=gradient_store_path,
            output_path=backward_result_path,
            input_ids=input_ids,
            targets=targets,
            device=device,
            parameter_working_set_bytes=parameter_working_set_bytes,
            gradient_working_set_bytes=gradient_working_set_bytes,
            activation_working_set_bytes=activation_working_set_bytes,
            workspace_working_set_bytes=workspace_working_set_bytes,
        )
        maximum_parameter_group_bytes = recomputed.maximum_parameter_group_bytes
        maximum_gradient_group_bytes = recomputed.maximum_gradient_group_bytes
        parameter_budget_respected = recomputed.parameter_budget_respected
        gradient_budget_respected = recomputed.gradient_budget_respected
        maximum_retained_activation_bytes = recomputed.maximum_retained_activation_bytes
        maximum_workspace_bytes = recomputed.maximum_workspace_bytes
        retained_forward_boundary_count = recomputed.retained_forward_boundary_count
        retained_forward_boundary_bytes = recomputed.retained_forward_boundary_bytes
        total_prefix_replayed_groups = recomputed.total_prefix_replayed_groups
        total_prefix_recomputation_seconds = recomputed.total_prefix_recomputation_seconds
        activation_budget_respected = recomputed.activation_budget_respected
        workspace_budget_respected = recomputed.workspace_budget_respected
        backward_resident_loss = recomputed.resident_loss
        bounded_loss = recomputed.recomputed_loss
        resident_gradient_norm = recomputed.resident_gradient_norm
        bounded_gradient_norm = recomputed.recomputed_gradient_norm
        clipping_coefficient = recomputed.future_clip_coefficient
    else:
        retained = run_bounded_backward_from_store(
            config,
            parameter_store=parameter_store,
            oracle_gradient_store_path=oracle_gradient_store_path,
            gradient_store_path=gradient_store_path,
            output_path=backward_result_path,
            input_ids=input_ids,
            targets=targets,
            device=device,
            parameter_working_set_bytes=parameter_working_set_bytes,
            gradient_working_set_bytes=gradient_working_set_bytes,
        )
        maximum_parameter_group_bytes = retained.maximum_parameter_group_bytes
        maximum_gradient_group_bytes = retained.maximum_gradient_group_bytes
        parameter_budget_respected = retained.parameter_budget_respected
        gradient_budget_respected = retained.gradient_budget_respected
        maximum_retained_activation_bytes = retained.retained_cpu_activation_bytes
        maximum_workspace_bytes = _retain_all_workspace_bytes(retained.backward_groups)
        retained_forward_boundary_count = max(0, len(retained.forward_groups) - 1)
        retained_forward_boundary_bytes = retained.retained_cpu_activation_bytes
        total_prefix_replayed_groups = 0
        total_prefix_recomputation_seconds = 0.0
        if maximum_retained_activation_bytes > activation_working_set_bytes:
            raise ActivationWorkingSetExceededError(
                "retain-all boundaries require "
                f"{maximum_retained_activation_bytes} activation bytes but the budget is "
                f"{activation_working_set_bytes} bytes"
            )
        if maximum_workspace_bytes > workspace_working_set_bytes:
            raise WorkspaceWorkingSetExceededError(
                "retain-all local backward requires "
                f"{maximum_workspace_bytes} workspace bytes but the budget is "
                f"{workspace_working_set_bytes} bytes"
            )
        activation_budget_respected = True
        workspace_budget_respected = True
        backward_resident_loss = retained.resident_loss
        bounded_loss = retained.bounded_loss
        resident_gradient_norm = retained.resident_gradient_norm
        bounded_gradient_norm = retained.bounded_gradient_norm
        clipping_coefficient = retained.future_clip_coefficient

    gradient_store = VersionedTensorStore.open(gradient_store_path)
    resident_loss, _ = _resident_oracle_from_current_state(
        config,
        parameter_store=parameter_store,
        optimizer_store=optimizer_store,
        oracle_store_path=oracle_state_store_path,
        input_ids=input_ids,
        targets=targets,
        device=device,
        clipping_coefficient=clipping_coefficient,
        committed_step=next_step,
    )
    if abs(resident_loss - backward_resident_loss) > 1e-5:
        raise RuntimeError("resident optimizer oracle loss differs from backward oracle loss")

    parameter_records = _parameter_records(parameter_store)
    gradient_records = _gradient_records(gradient_store)
    optimizer_records, param_groups_record = _optimizer_records(optimizer_store)
    execution_groups = build_execution_groups(config, set(parameter_records))
    groups = _optimizer_groups(execution_groups)
    candidate_parameter_store = VersionedTensorStore.create(
        candidate_parameter_store_path,
        limits=_store_limits(config),
    )
    candidate_optimizer_store = VersionedTensorStore.create(
        candidate_optimizer_store_path,
        limits=_store_limits(config),
    )

    maximum_optimizer_group_bytes = 0
    optimizer_metrics: list[OptimizerGroupMetrics] = []
    tied_updates = 0
    param_groups_payload = optimizer_store.read_tensor(param_groups_record.tensor_id)
    for group in groups:
        parameter_group_records = tuple(parameter_records[name] for name in group.tensor_names)
        gradient_group_records = tuple(gradient_records[name] for name in group.tensor_names)
        bare_names = tuple(_bare_parameter_name(name) for name in group.tensor_names)
        state_group_records = {
            bare_name: optimizer_records[bare_name] for bare_name in bare_names
        }
        parameter_bytes = sum(item.byte_length for item in parameter_group_records)
        gradient_bytes = sum(item.byte_length for item in gradient_group_records)
        first_moment_bytes = sum(
            state_group_records[name]["exp_avg"].byte_length for name in bare_names
        )
        second_moment_bytes = sum(
            state_group_records[name]["exp_avg_sq"].byte_length for name in bare_names
        )
        step_bytes = sum(
            state_group_records[name]["step"].byte_length for name in bare_names
        )
        logical_working_set = (
            parameter_bytes
            + gradient_bytes
            + first_moment_bytes
            + second_moment_bytes
            + step_bytes
        )
        maximum_optimizer_group_bytes = max(
            maximum_optimizer_group_bytes,
            logical_working_set,
        )
        if logical_working_set > optimizer_working_set_bytes:
            raise OptimizerWorkingSetExceededError(
                f"optimizer group {group.name} requires {logical_working_set} bytes"
            )

        parameter_payloads, parameter_read_seconds = _read_payloads(
            parameter_store,
            parameter_group_records,
        )
        gradient_payloads, gradient_read_seconds = _read_payloads(
            gradient_store,
            gradient_group_records,
        )
        optimizer_read_started = time.perf_counter()
        optimizer_payloads: dict[str, dict[str, TensorPayload]] = {}
        for bare_name in bare_names:
            optimizer_payloads[bare_name] = {
                state_key: optimizer_store.read_tensor(record.tensor_id)
                for state_key, record in state_group_records[bare_name].items()
            }
        optimizer_state_read_seconds = time.perf_counter() - optimizer_read_started
        (
            updated_parameters,
            updated_optimizer,
            materialization_seconds,
            optimizer_seconds,
            export_seconds,
        ) = _apply_adamw_group(
            config=config,
            parameter_payloads=parameter_payloads,
            gradient_payloads=gradient_payloads,
            optimizer_payloads=optimizer_payloads,
            clipping_coefficient=clipping_coefficient,
            device=device,
        )
        accelerator_after_optimizer = accelerator_memory_metrics(device)
        process_rss = process_rss_bytes()
        updated_parameter_checksum = _payload_digest(updated_parameters)
        updated_optimizer_checksum = _payload_digest(updated_optimizer)

        parameter_commit_started = time.perf_counter()
        parameter_commit = _commit_versioned_payloads(
            candidate_parameter_store,
            updated_parameters,
            committed_step=group.ordinal + 1,
            version=next_step,
        )
        parameter_commit_seconds = time.perf_counter() - parameter_commit_started
        optimizer_commit_started = time.perf_counter()
        optimizer_commit_payloads = updated_optimizer
        if group.ordinal == 0:
            optimizer_commit_payloads = tuple(updated_optimizer) + (param_groups_payload,)
        optimizer_commit = _commit_versioned_payloads(
            candidate_optimizer_store,
            optimizer_commit_payloads,
            committed_step=group.ordinal + 1,
            version=next_step,
        )
        optimizer_commit_seconds = time.perf_counter() - optimizer_commit_started
        if "model.token_embedding.weight" in group.tensor_names:
            tied_updates += 1

        release_started = time.perf_counter()
        del (
            parameter_payloads,
            gradient_payloads,
            optimizer_payloads,
            updated_parameters,
            updated_optimizer,
            optimizer_commit_payloads,
        )
        gc.collect()
        synchronize_accelerator(device)
        release_seconds = time.perf_counter() - release_started
        accelerator_after_release = accelerator_memory_metrics(device)
        optimizer_metrics.append(
            OptimizerGroupMetrics(
                ordinal=group.ordinal,
                name=group.name,
                parameter_names=group.tensor_names,
                parameter_bytes=parameter_bytes,
                gradient_bytes=gradient_bytes,
                first_moment_bytes=first_moment_bytes,
                second_moment_bytes=second_moment_bytes,
                step_bytes=step_bytes,
                logical_working_set_bytes=logical_working_set,
                parameter_read_seconds=parameter_read_seconds,
                gradient_read_seconds=gradient_read_seconds,
                optimizer_state_read_seconds=optimizer_state_read_seconds,
                materialization_seconds=materialization_seconds,
                optimizer_seconds=optimizer_seconds,
                export_seconds=export_seconds,
                parameter_commit_seconds=parameter_commit_seconds,
                optimizer_commit_seconds=optimizer_commit_seconds,
                release_seconds=release_seconds,
                parameter_logical_bytes_written=parameter_commit.logical_bytes_written,
                parameter_physical_bytes_written=parameter_commit.physical_bytes_written,
                parameter_chunk_writes=parameter_commit.chunk_writes,
                parameter_chunks_reused=parameter_commit.chunks_reused,
                optimizer_logical_bytes_written=optimizer_commit.logical_bytes_written,
                optimizer_physical_bytes_written=optimizer_commit.physical_bytes_written,
                optimizer_chunk_writes=optimizer_commit.chunk_writes,
                optimizer_chunks_reused=optimizer_commit.chunks_reused,
                updated_parameter_checksum=updated_parameter_checksum,
                updated_optimizer_checksum=updated_optimizer_checksum,
                process_rss_after_optimizer_bytes=process_rss,
                accelerator_after_optimizer=accelerator_after_optimizer,
                accelerator_after_release=accelerator_after_release,
            )
        )

    candidate_parameter_store.verify()
    candidate_optimizer_store.verify()
    oracle_state_store = VersionedTensorStore.open(oracle_state_store_path)
    oracle_state = _store_payloads(oracle_state_store)
    candidate_state = tuple(
        sorted(
            _store_payloads(candidate_parameter_store)
            + _store_payloads(candidate_optimizer_store),
            key=lambda item: item.logical_name,
        )
    )
    resident_vs_candidate = compare_states(oracle_state, candidate_state)

    restored_model = DecoderOnlyTransformer(config.model).to(device)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=False,
    )
    restore_pytorch_state(restored_model, candidate_state, optimizer=restored_optimizer)
    restored_state = export_pytorch_state(restored_model, restored_optimizer)
    candidate_vs_restored = compare_states(candidate_state, restored_state)
    del restored_state, restored_optimizer, restored_model
    gc.collect()
    synchronize_accelerator(device)

    pre_publish_current = bundle_store.current_manifest()
    final_bundle, final_publication = bundle_store.publish(
        committed_step=next_step,
        parameter_store_path=candidate_parameter_store_path,
        optimizer_store_path=candidate_optimizer_store_path,
        gradient_store_path=gradient_store_path,
        batch_checksum=data_checksum,
        failure_injector=bundle_failure_injector,
    )
    final_verification = bundle_store.verify(final_bundle.bundle_id)
    final_current = bundle_store.current_manifest()
    candidate_versions = tuple(
        sorted(
            (record.tensor_id, record.version)
            for record in (
                candidate_parameter_store.current_manifest().tensors
                + candidate_optimizer_store.current_manifest().tensors
            )
        )
    )
    result = PersistentStepResult(
        step=next_step,
        batch_cursor=batch_cursor,
        batch_seed=batch_seed,
        batch_checksum=data_checksum,
        source_bundle_id=current.bundle_id,
        final_bundle_id=final_bundle.bundle_id,
        source_parameter_store_path=str(parameter_store.root),
        source_optimizer_store_path=str(optimizer_store.root),
        gradient_store_path=str(gradient_store_path),
        candidate_parameter_store_path=str(candidate_parameter_store_path),
        candidate_optimizer_store_path=str(candidate_optimizer_store_path),
        oracle_state_store_path=str(oracle_state_store_path),
        bounded_backward_result_path=str(backward_result_path),
        activation_policy=config.training.activation_policy,
        parameter_working_set_budget_bytes=parameter_working_set_bytes,
        gradient_working_set_budget_bytes=gradient_working_set_bytes,
        optimizer_working_set_budget_bytes=optimizer_working_set_bytes,
        activation_working_set_budget_bytes=activation_working_set_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_bytes,
        maximum_parameter_group_bytes=maximum_parameter_group_bytes,
        maximum_gradient_group_bytes=maximum_gradient_group_bytes,
        maximum_optimizer_group_bytes=maximum_optimizer_group_bytes,
        maximum_retained_activation_bytes=maximum_retained_activation_bytes,
        maximum_workspace_bytes=maximum_workspace_bytes,
        retained_forward_boundary_count=retained_forward_boundary_count,
        retained_forward_boundary_bytes=retained_forward_boundary_bytes,
        total_prefix_replayed_groups=total_prefix_replayed_groups,
        total_prefix_recomputation_seconds=total_prefix_recomputation_seconds,
        parameter_budget_respected=parameter_budget_respected,
        gradient_budget_respected=gradient_budget_respected,
        optimizer_budget_respected=maximum_optimizer_group_bytes
        <= optimizer_working_set_bytes,
        activation_budget_respected=activation_budget_respected,
        workspace_budget_respected=workspace_budget_respected,
        resident_loss=resident_loss,
        bounded_loss=bounded_loss,
        resident_gradient_norm=resident_gradient_norm,
        loss_absolute_difference=abs(resident_loss - bounded_loss),
        bounded_gradient_norm=bounded_gradient_norm,
        gradient_norm_absolute_difference=abs(
            resident_gradient_norm - bounded_gradient_norm
        ),
        clipping_coefficient=clipping_coefficient,
        optimizer_group_order=tuple(item.name for item in optimizer_metrics),
        optimizer_groups=tuple(optimizer_metrics),
        tied_parameter_update_count=tied_updates,
        candidate_tensor_versions=candidate_versions,
        resident_vs_candidate_state=resident_vs_candidate,
        candidate_vs_restored_state=candidate_vs_restored,
        source_bundle_remained_authoritative_until_final_publish=(
            pre_publish_current.bundle_id == current.bundle_id
        ),
        final_bundle_is_authoritative=final_current.bundle_id == final_bundle.bundle_id,
        final_bundle_publication=final_publication,
        final_bundle_verification=final_verification,
        total_parameter_logical_bytes_read=sum(
            item.parameter_bytes for item in optimizer_metrics
        ),
        total_gradient_logical_bytes_read=sum(
            item.gradient_bytes for item in optimizer_metrics
        ),
        total_optimizer_logical_bytes_read=sum(
            item.first_moment_bytes + item.second_moment_bytes + item.step_bytes
            for item in optimizer_metrics
        ),
        total_parameter_logical_bytes_written=sum(
            item.parameter_logical_bytes_written for item in optimizer_metrics
        ),
        total_parameter_physical_bytes_written=sum(
            item.parameter_physical_bytes_written for item in optimizer_metrics
        ),
        total_optimizer_logical_bytes_written=sum(
            item.optimizer_logical_bytes_written for item in optimizer_metrics
        ),
        total_optimizer_physical_bytes_written=sum(
            item.optimizer_physical_bytes_written for item in optimizer_metrics
        ),
    )
    del oracle_state, candidate_state
    gc.collect()
    synchronize_accelerator(device)
    return result
