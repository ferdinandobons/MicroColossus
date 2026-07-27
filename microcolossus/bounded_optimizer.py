"""Group-bounded AdamW execution with atomic step-bundle publication."""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .bounded_backward import run_bounded_backward
from .bounded_forward import ExecutionGroupSpec, build_execution_groups
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .step_bundle import (
    BundleFailureInjector,
    BundlePublicationTelemetry,
    BundleVerificationReport,
    StepBundleStore,
)
from .storage import StoreLimits, TensorKind, TensorPayload, VersionedTensorStore
from .storage.adapters import (
    export_pytorch_adamw,
    export_pytorch_state,
    payload_from_torch,
    payload_to_torch,
    restore_pytorch_model,
    restore_pytorch_state,
)
from .storage.schema import StoreTelemetry, TensorRecord
from .storage_training import StateComparison, compare_states
from .telemetry import (
    AcceleratorMemoryMetrics,
    accelerator_memory_metrics,
    process_rss_bytes,
    synchronize_accelerator,
    write_json_atomic,
)
from .training import make_synthetic_lm_batch, resolve_device, run_resident_step, seed_everything

BOUNDED_OPTIMIZER_SCHEMA_VERSION = "microcolossus.bounded-optimizer.v1"


class OptimizerWorkingSetExceededError(RuntimeError):
    """Raised when a complete parameter, gradient, and Adam group exceeds its budget."""


@dataclass(frozen=True)
class OptimizerGroupMetrics:
    ordinal: int
    name: str
    parameter_names: tuple[str, ...]
    parameter_bytes: int
    gradient_bytes: int
    first_moment_bytes: int
    second_moment_bytes: int
    step_bytes: int
    logical_working_set_bytes: int
    parameter_read_seconds: float
    gradient_read_seconds: float
    optimizer_state_read_seconds: float
    materialization_seconds: float
    optimizer_seconds: float
    export_seconds: float
    parameter_commit_seconds: float
    optimizer_commit_seconds: float
    release_seconds: float
    parameter_logical_bytes_written: int
    parameter_physical_bytes_written: int
    parameter_chunk_writes: int
    parameter_chunks_reused: int
    optimizer_logical_bytes_written: int
    optimizer_physical_bytes_written: int
    optimizer_chunk_writes: int
    optimizer_chunks_reused: int
    updated_parameter_checksum: str
    updated_optimizer_checksum: str
    process_rss_after_optimizer_bytes: int
    accelerator_after_optimizer: AcceleratorMemoryMetrics
    accelerator_after_release: AcceleratorMemoryMetrics


@dataclass(frozen=True)
class ResidentOptimizerTrace:
    loss: float
    gradient_norm: float
    parameter_checksum: str
    store_commit: StoreTelemetry


@dataclass(frozen=True)
class BoundedOptimizerResult:
    schema_version: str
    experiment: str
    device: str
    bundle_store_path: str
    bounded_backward_result_path: str
    parameter_count: int
    batch_checksum: str
    initial_bundle_id: str
    final_bundle_id: str
    initial_bundle_step: int
    final_bundle_step: int
    parameter_store_path: str
    initial_optimizer_store_path: str
    oracle_state_store_path: str
    candidate_parameter_store_path: str
    candidate_optimizer_store_path: str
    gradient_store_path: str
    parameter_working_set_budget_bytes: int
    gradient_working_set_budget_bytes: int
    optimizer_working_set_budget_bytes: int
    maximum_parameter_group_bytes: int
    maximum_gradient_group_bytes: int
    maximum_optimizer_group_bytes: int
    parameter_budget_respected: bool
    gradient_budget_respected: bool
    optimizer_budget_respected: bool
    resident_oracle_released_before_streamed_optimizer: bool
    initial_optimizer_payloads_released_before_streamed_optimizer: bool
    full_candidate_state_materialized_for_validation: bool
    resident_loss: float
    bounded_loss: float
    loss_absolute_difference: float
    resident_gradient_norm: float
    bounded_gradient_norm: float
    gradient_norm_absolute_difference: float
    clipping_coefficient: float
    optimizer_group_order: tuple[str, ...]
    optimizer_groups: tuple[OptimizerGroupMetrics, ...]
    tied_parameter_update_count: int
    candidate_tensor_versions: tuple[tuple[str, int], ...]
    resident_vs_candidate_state: StateComparison
    candidate_vs_restored_state: StateComparison
    initial_bundle_publication: BundlePublicationTelemetry
    final_bundle_publication: BundlePublicationTelemetry
    final_bundle_verification: BundleVerificationReport
    initial_bundle_remained_authoritative_until_final_publish: bool
    final_bundle_is_authoritative: bool
    total_parameter_logical_bytes_read: int
    total_gradient_logical_bytes_read: int
    total_optimizer_logical_bytes_read: int
    total_parameter_logical_bytes_written: int
    total_parameter_physical_bytes_written: int
    total_optimizer_logical_bytes_written: int
    total_optimizer_physical_bytes_written: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _store_limits(config: ExperimentConfig) -> StoreLimits:
    mib = 1024**2
    return StoreLimits(
        chunk_size_bytes=mib,
        max_storage_bytes=int(config.hardware.nvme_gib * 1024**3),
        max_staging_bytes=16 * mib,
    )


def _store_payloads(store: VersionedTensorStore) -> tuple[TensorPayload, ...]:
    return tuple(
        store.read_tensor(record.tensor_id)
        for record in sorted(
            store.current_manifest().tensors,
            key=lambda item: item.logical_name,
        )
    )


def _payload_digest(payloads: tuple[TensorPayload, ...]) -> str:
    digest = hashlib.sha256()
    for payload in sorted(payloads, key=lambda item: item.logical_name):
        digest.update(payload.logical_name.encode("utf-8"))
        digest.update(payload.checksum.encode("ascii"))
    return digest.hexdigest()


def _initialize_adamw_payloads(config: ExperimentConfig) -> tuple[TensorPayload, ...]:
    model = DecoderOnlyTransformer(config.model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.zeros_like(parameter)
    payloads = export_pytorch_adamw(model, optimizer)
    del optimizer, model
    gc.collect()
    return payloads


def _parameter_records(store: VersionedTensorStore) -> dict[str, TensorRecord]:
    return {
        record.logical_name: record
        for record in store.current_manifest().tensors
        if record.kind is TensorKind.PARAMETER
    }


def _gradient_records(store: VersionedTensorStore) -> dict[str, TensorRecord]:
    result: dict[str, TensorRecord] = {}
    for record in store.current_manifest().tensors:
        parameter_name = dict(record.metadata).get("parameter_name")
        if parameter_name is not None:
            result[parameter_name] = record
    return result


def _optimizer_records(
    store: VersionedTensorStore,
) -> tuple[dict[str, dict[str, TensorRecord]], TensorRecord]:
    result: dict[str, dict[str, TensorRecord]] = {}
    param_groups: TensorRecord | None = None
    for record in store.current_manifest().tensors:
        metadata = dict(record.metadata)
        if metadata.get("record_type") == "param_groups":
            param_groups = record
            continue
        parameter_name = metadata.get("parameter_name")
        state_key = metadata.get("state_key")
        if parameter_name is None or state_key is None:
            continue
        result.setdefault(parameter_name, {})[state_key] = record
    if param_groups is None:
        raise KeyError("optimizer param_groups metadata is missing")
    return result, param_groups


def _optimizer_groups(groups: tuple[ExecutionGroupSpec, ...]) -> tuple[ExecutionGroupSpec, ...]:
    seen: set[str] = set()
    result: list[ExecutionGroupSpec] = []
    for group in groups:
        unique = tuple(name for name in group.tensor_names if name not in seen)
        seen.update(unique)
        if unique:
            result.append(
                ExecutionGroupSpec(
                    ordinal=len(result),
                    name=group.name,
                    tensor_names=unique,
                )
            )
    return tuple(result)


def _bare_parameter_name(logical_name: str) -> str:
    if not logical_name.startswith("model."):
        raise ValueError(f"expected model parameter logical name: {logical_name}")
    return logical_name.removeprefix("model.")


def _read_payloads(
    store: VersionedTensorStore,
    records: tuple[TensorRecord, ...],
) -> tuple[tuple[TensorPayload, ...], float]:
    started = time.perf_counter()
    payloads = tuple(store.read_tensor(record.tensor_id) for record in records)
    return payloads, time.perf_counter() - started


def _commit_payloads(
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


def _resident_oracle(
    config: ExperimentConfig,
    parameter_store: VersionedTensorStore,
    oracle_store_path: Path,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
) -> ResidentOptimizerTrace:
    model = DecoderOnlyTransformer(config.model).to(device)
    restore_pytorch_model(
        model,
        (
            parameter_store.read_tensor(record.tensor_id)
            for record in sorted(
                parameter_store.current_manifest().tensors,
                key=lambda item: item.logical_name,
            )
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    metrics = run_resident_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        device=device,
        step=0,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )
    state_payloads = export_pytorch_state(model, optimizer)
    oracle_store = VersionedTensorStore.create(
        oracle_store_path,
        limits=_store_limits(config),
    )
    transaction = oracle_store.begin_transaction(committed_step=1)
    transaction.put_many(state_payloads)
    commit = transaction.commit()
    del state_payloads, optimizer, model
    gc.collect()
    synchronize_accelerator(device)
    return ResidentOptimizerTrace(
        loss=metrics.loss,
        gradient_norm=metrics.gradient_norm,
        parameter_checksum=metrics.parameter_checksum,
        store_commit=commit.telemetry,
    )


def _apply_adamw_group(
    *,
    config: ExperimentConfig,
    parameter_payloads: tuple[TensorPayload, ...],
    gradient_payloads: tuple[TensorPayload, ...],
    optimizer_payloads: dict[str, dict[str, TensorPayload]],
    clipping_coefficient: float,
    device: torch.device,
) -> tuple[tuple[TensorPayload, ...], tuple[TensorPayload, ...]]:
    parameters: list[nn.Parameter] = []
    names: list[str] = []
    parameter_payload_map = {item.logical_name: item for item in parameter_payloads}
    gradient_payload_map = {
        dict(item.metadata)["parameter_name"]: item for item in gradient_payloads
    }
    for logical_name in sorted(parameter_payload_map):
        payload = parameter_payload_map[logical_name]
        parameter = nn.Parameter(payload_to_torch(payload, device=device))
        gradient = payload_to_torch(gradient_payload_map[logical_name], device=device)
        parameter.grad = gradient.mul(clipping_coefficient)
        parameters.append(parameter)
        names.append(logical_name)

    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    for logical_name, parameter in zip(names, parameters, strict=True):
        bare_name = _bare_parameter_name(logical_name)
        state_payloads = optimizer_payloads[bare_name]
        optimizer.state[parameter]["step"] = payload_to_torch(
            state_payloads["step"], device="cpu"
        )
        optimizer.state[parameter]["exp_avg"] = payload_to_torch(
            state_payloads["exp_avg"], device=device
        )
        optimizer.state[parameter]["exp_avg_sq"] = payload_to_torch(
            state_payloads["exp_avg_sq"], device=device
        )
    optimizer.step()
    synchronize_accelerator(device)

    updated_parameters: list[TensorPayload] = []
    updated_optimizer: list[TensorPayload] = []
    for logical_name, parameter in zip(names, parameters, strict=True):
        original = parameter_payload_map[logical_name]
        updated_parameters.append(
            payload_from_torch(
                parameter,
                logical_name=original.logical_name,
                kind=original.kind,
                metadata=original.metadata,
            )
        )
        bare_name = _bare_parameter_name(logical_name)
        for state_key in ("step", "exp_avg", "exp_avg_sq"):
            original_state = optimizer_payloads[bare_name][state_key]
            updated_optimizer.append(
                payload_from_torch(
                    optimizer.state[parameter][state_key],
                    logical_name=original_state.logical_name,
                    kind=original_state.kind,
                    metadata=original_state.metadata,
                )
            )
    return (
        tuple(sorted(updated_parameters, key=lambda item: item.logical_name)),
        tuple(sorted(updated_optimizer, key=lambda item: item.logical_name)),
    )


def run_bounded_optimizer_step(
    config: ExperimentConfig,
    *,
    bundle_store_path: str | Path,
    output_path: str | Path | None = None,
    device_override: str | None = None,
    parameter_working_set_bytes: int = 1024**2,
    gradient_working_set_bytes: int = 1024**2,
    optimizer_working_set_bytes: int = 4 * 1024**2,
    bundle_failure_injector: BundleFailureInjector | None = None,
) -> BoundedOptimizerResult:
    """Execute one bounded AdamW step and atomically publish its state bundle."""

    if optimizer_working_set_bytes <= 0:
        raise ValueError("optimizer_working_set_bytes must be greater than zero")
    destination = Path(bundle_store_path)
    bundle_store = StepBundleStore.create(destination)
    work = destination / "work"
    candidates = destination / "candidates"
    backward_result_path = work / "bounded-backward.json"
    parameter_store_path = work / "parameters"
    oracle_gradient_store_path = work / "oracle-gradients"
    gradient_store_path = work / "bounded-gradients"

    backward = run_bounded_backward(
        config,
        parameter_store_path=parameter_store_path,
        oracle_gradient_store_path=oracle_gradient_store_path,
        gradient_store_path=gradient_store_path,
        output_path=backward_result_path,
        device_override=device_override,
        parameter_working_set_bytes=parameter_working_set_bytes,
        gradient_working_set_bytes=gradient_working_set_bytes,
    )
    device = resolve_device(device_override or config.training.device)
    parameter_store = VersionedTensorStore.open(parameter_store_path)
    gradient_store = VersionedTensorStore.open(gradient_store_path)

    initial_optimizer_store_path = work / "initial-optimizer"
    initial_optimizer_store = VersionedTensorStore.create(
        initial_optimizer_store_path,
        limits=_store_limits(config),
    )
    initial_optimizer_payloads = _initialize_adamw_payloads(config)
    initial_optimizer_transaction = initial_optimizer_store.begin_transaction(committed_step=0)
    initial_optimizer_transaction.put_many(initial_optimizer_payloads)
    initial_optimizer_transaction.commit()

    initial_bundle, initial_publication = bundle_store.publish(
        committed_step=0,
        parameter_store_path=parameter_store_path,
        optimizer_store_path=initial_optimizer_store_path,
        gradient_store_path=None,
        batch_checksum=backward.batch_checksum,
    )

    generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 1)
    input_ids, targets = make_synthetic_lm_batch(
        batch_size=config.training.micro_batch_size,
        sequence_length=config.training.sequence_length,
        vocab_size=config.model.vocab_size,
        generator=generator,
    )
    oracle_state_store_path = work / "oracle-state"
    resident = _resident_oracle(
        config,
        parameter_store,
        oracle_state_store_path,
        input_ids,
        targets,
        device,
    )
    del initial_optimizer_payloads
    gc.collect()
    synchronize_accelerator(device)

    parameter_records = _parameter_records(parameter_store)
    gradient_records = _gradient_records(gradient_store)
    optimizer_records, param_groups_record = _optimizer_records(initial_optimizer_store)
    execution_groups = build_execution_groups(config, set(parameter_records))
    groups = _optimizer_groups(execution_groups)

    candidate_parameter_store_path = candidates / "step-1-parameters"
    candidate_optimizer_store_path = candidates / "step-1-optimizer"
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
    param_groups_payload = initial_optimizer_store.read_tensor(param_groups_record.tensor_id)
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
        step_bytes = sum(state_group_records[name]["step"].byte_length for name in bare_names)
        logical_working_set = (
            parameter_bytes
            + gradient_bytes
            + first_moment_bytes
            + second_moment_bytes
            + step_bytes
        )
        maximum_optimizer_group_bytes = max(
            maximum_optimizer_group_bytes, logical_working_set
        )
        if logical_working_set > optimizer_working_set_bytes:
            raise OptimizerWorkingSetExceededError(
                f"optimizer group {group.name} requires {logical_working_set} bytes"
            )

        parameter_payloads, parameter_read_seconds = _read_payloads(
            parameter_store, parameter_group_records
        )
        gradient_payloads, gradient_read_seconds = _read_payloads(
            gradient_store, gradient_group_records
        )
        optimizer_read_started = time.perf_counter()
        optimizer_payloads: dict[str, dict[str, TensorPayload]] = {}
        for bare_name in bare_names:
            optimizer_payloads[bare_name] = {
                state_key: initial_optimizer_store.read_tensor(record.tensor_id)
                for state_key, record in state_group_records[bare_name].items()
            }
        optimizer_state_read_seconds = time.perf_counter() - optimizer_read_started

        materialization_started = time.perf_counter()
        materialization_seconds = time.perf_counter() - materialization_started
        optimizer_started = time.perf_counter()
        updated_parameters, updated_optimizer = _apply_adamw_group(
            config=config,
            parameter_payloads=parameter_payloads,
            gradient_payloads=gradient_payloads,
            optimizer_payloads=optimizer_payloads,
            clipping_coefficient=backward.future_clip_coefficient,
            device=device,
        )
        optimizer_seconds = time.perf_counter() - optimizer_started
        accelerator_after_optimizer = accelerator_memory_metrics(device)
        process_rss = process_rss_bytes()

        export_started = time.perf_counter()
        updated_parameter_checksum = _payload_digest(updated_parameters)
        updated_optimizer_checksum = _payload_digest(updated_optimizer)
        export_seconds = time.perf_counter() - export_started

        parameter_commit_started = time.perf_counter()
        parameter_commit = _commit_payloads(
            candidate_parameter_store,
            updated_parameters,
            committed_step=group.ordinal + 1,
            version=1,
        )
        parameter_commit_seconds = time.perf_counter() - parameter_commit_started
        optimizer_commit_started = time.perf_counter()
        optimizer_commit_payloads = updated_optimizer
        if group.ordinal == 0:
            optimizer_commit_payloads = tuple(updated_optimizer) + (param_groups_payload,)
        optimizer_commit = _commit_payloads(
            candidate_optimizer_store,
            optimizer_commit_payloads,
            committed_step=group.ordinal + 1,
            version=1,
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
    pre_publish_current = bundle_store.current_manifest()
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
    )
    restore_pytorch_state(restored_model, candidate_state, optimizer=restored_optimizer)
    restored_state = export_pytorch_state(restored_model, restored_optimizer)
    candidate_vs_restored = compare_states(candidate_state, restored_state)
    del restored_state, restored_optimizer, restored_model
    gc.collect()
    synchronize_accelerator(device)

    final_bundle, final_publication = bundle_store.publish(
        committed_step=1,
        parameter_store_path=candidate_parameter_store_path,
        optimizer_store_path=candidate_optimizer_store_path,
        gradient_store_path=gradient_store_path,
        batch_checksum=backward.batch_checksum,
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
    result = BoundedOptimizerResult(
        schema_version=BOUNDED_OPTIMIZER_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        bundle_store_path=str(destination),
        bounded_backward_result_path=str(backward_result_path),
        parameter_count=backward.parameter_count,
        batch_checksum=backward.batch_checksum,
        initial_bundle_id=initial_bundle.bundle_id,
        final_bundle_id=final_bundle.bundle_id,
        initial_bundle_step=initial_bundle.committed_step,
        final_bundle_step=final_bundle.committed_step,
        parameter_store_path=str(parameter_store_path),
        initial_optimizer_store_path=str(initial_optimizer_store_path),
        oracle_state_store_path=str(oracle_state_store_path),
        candidate_parameter_store_path=str(candidate_parameter_store_path),
        candidate_optimizer_store_path=str(candidate_optimizer_store_path),
        gradient_store_path=str(gradient_store_path),
        parameter_working_set_budget_bytes=parameter_working_set_bytes,
        gradient_working_set_budget_bytes=gradient_working_set_bytes,
        optimizer_working_set_budget_bytes=optimizer_working_set_bytes,
        maximum_parameter_group_bytes=backward.maximum_parameter_group_bytes,
        maximum_gradient_group_bytes=backward.maximum_gradient_group_bytes,
        maximum_optimizer_group_bytes=maximum_optimizer_group_bytes,
        parameter_budget_respected=backward.parameter_budget_respected,
        gradient_budget_respected=backward.gradient_budget_respected,
        optimizer_budget_respected=(
            maximum_optimizer_group_bytes <= optimizer_working_set_bytes
        ),
        resident_oracle_released_before_streamed_optimizer=True,
        initial_optimizer_payloads_released_before_streamed_optimizer=True,
        full_candidate_state_materialized_for_validation=True,
        resident_loss=resident.loss,
        bounded_loss=backward.bounded_loss,
        loss_absolute_difference=abs(resident.loss - backward.bounded_loss),
        resident_gradient_norm=resident.gradient_norm,
        bounded_gradient_norm=backward.bounded_gradient_norm,
        gradient_norm_absolute_difference=abs(
            resident.gradient_norm - backward.bounded_gradient_norm
        ),
        clipping_coefficient=backward.future_clip_coefficient,
        optimizer_group_order=tuple(item.name for item in optimizer_metrics),
        optimizer_groups=tuple(optimizer_metrics),
        tied_parameter_update_count=tied_updates,
        candidate_tensor_versions=candidate_versions,
        resident_vs_candidate_state=resident_vs_candidate,
        candidate_vs_restored_state=candidate_vs_restored,
        initial_bundle_publication=initial_publication,
        final_bundle_publication=final_publication,
        final_bundle_verification=final_verification,
        initial_bundle_remained_authoritative_until_final_publish=(
            pre_publish_current.bundle_id == initial_bundle.bundle_id
        ),
        final_bundle_is_authoritative=(final_current.bundle_id == final_bundle.bundle_id),
        total_parameter_logical_bytes_read=sum(
            item.parameter_bytes for item in optimizer_metrics
        ),
        total_gradient_logical_bytes_read=sum(item.gradient_bytes for item in optimizer_metrics),
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
    if output_path is not None:
        write_json_atomic(output_path, result)
    del oracle_state, candidate_state
    gc.collect()
    synchronize_accelerator(device)
    return result
