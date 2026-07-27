"""Persistent multi-step bounded training with checkpoint and resume."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import torch

from .bounded_optimizer import _optimizer_records, _store_payloads
from .bounded_training_types import BoundedTrainingResult, PersistentStepResult
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .persistent_step import advance_one_step
from .step_bundle import BundleFailureInjector, BundlePublicationTelemetry, StepBundleStore
from .storage import IntegrityError, TensorPayload
from .storage.adapters import export_pytorch_state, payload_to_torch, restore_pytorch_state
from .storage_training import StateComparison, compare_states
from .telemetry import synchronize_accelerator, write_json_atomic
from .training import resolve_device, seed_everything
from .training_checkpoint import (
    BATCH_STREAM_VERSION,
    BOUNDED_TRAINING_SCHEMA_VERSION,
    MULTI_STEP_RUNTIME_VERSION,
    ResumeConfigurationError,
    _batch_for_cursor,
    _initialize_training_root,
    _lineage,
    _load_training_metadata,
    _open_referenced_store,
    _validate_resume_metadata,
)

__all__ = [
    "BATCH_STREAM_VERSION",
    "BOUNDED_TRAINING_SCHEMA_VERSION",
    "MULTI_STEP_RUNTIME_VERSION",
    "BoundedTrainingResult",
    "ResumeConfigurationError",
    "run_bounded_training",
]


def _resident_reference_state(
    config: ExperimentConfig,
    *,
    target_step: int,
    device: torch.device,
) -> tuple[TensorPayload, ...]:
    seed_everything(config.training.seed)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=False,
    )
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.zeros_like(parameter)
    model.train()
    for cursor in range(target_step):
        input_ids, targets, _ = _batch_for_cursor(config, cursor)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids.to(device), targets.to(device))
        if output.loss is None:
            raise RuntimeError("resident reference did not return a loss")
        output.loss.backward()
        squared_norms: list[float] = []
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norms.append(
                    float(parameter.grad.detach().float().pow(2).sum().cpu().item())
                )
        global_norm = math.sqrt(math.fsum(squared_norms))
        max_norm = config.training.gradient_clip_norm
        coefficient = (
            1.0 if max_norm is None else min(1.0, max_norm / (global_norm + 1e-6))
        )
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
        synchronize_accelerator(device)
        optimizer.step()
        synchronize_accelerator(device)
        del output
    state = export_pytorch_state(model, optimizer)
    del optimizer, model
    gc.collect()
    synchronize_accelerator(device)
    return state


def _current_bundle_state(
    config: ExperimentConfig,
    bundle_store: StepBundleStore,
    device: torch.device,
) -> tuple[tuple[TensorPayload, ...], StateComparison, tuple[tuple[str, float], ...]]:
    current = bundle_store.current_manifest()
    parameter_store = _open_referenced_store(bundle_store, current, kind="parameter")
    optimizer_store = _open_referenced_store(bundle_store, current, kind="optimizer")
    state = tuple(
        sorted(
            _store_payloads(parameter_store) + _store_payloads(optimizer_store),
            key=lambda item: item.logical_name,
        )
    )
    restored_model = DecoderOnlyTransformer(config.model).to(device)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=False,
    )
    restore_pytorch_state(restored_model, state, optimizer=restored_optimizer)
    restored = export_pytorch_state(restored_model, restored_optimizer)
    comparison = compare_states(state, restored)
    optimizer_records, _ = _optimizer_records(optimizer_store)
    step_values: list[tuple[str, float]] = []
    for parameter_name, records in sorted(optimizer_records.items()):
        payload = optimizer_store.read_tensor(records["step"].tensor_id)
        value = payload_to_torch(payload)
        step_values.append((parameter_name, float(value.item())))
    del restored, restored_optimizer, restored_model
    gc.collect()
    synchronize_accelerator(device)
    return state, comparison, tuple(step_values)


def run_bounded_training(
    config: ExperimentConfig,
    *,
    bundle_store_path: str | Path,
    target_step: int,
    output_path: str | Path | None = None,
    device_override: str | None = None,
    parameter_working_set_bytes: int = 1024**2,
    gradient_working_set_bytes: int = 1024**2,
    optimizer_working_set_bytes: int = 4 * 1024**2,
    bundle_failure_injector: BundleFailureInjector | None = None,
) -> BoundedTrainingResult:
    """Advance a persistent bounded training root to ``target_step``."""

    if target_step < 0:
        raise ValueError("target_step cannot be negative")
    destination = Path(bundle_store_path)
    initialization_publication: BundlePublicationTelemetry | None = None
    if destination.exists():
        bundle_store = StepBundleStore.open(destination)
        metadata = _load_training_metadata(destination)
        _validate_resume_metadata(config, metadata)
        current = bundle_store.current_manifest()
        resumed = True
        initialized_bundle_id = _lineage(bundle_store)[0].bundle_id
    else:
        bundle_store, current, initialization_publication, metadata = (
            _initialize_training_root(config, destination)
        )
        resumed = False
        initialized_bundle_id = current.bundle_id
    bundle_store.verify(current.bundle_id)
    started_step = current.committed_step
    if target_step < started_step:
        raise ValueError(
            f"target_step {target_step} is behind current committed step {started_step}"
        )

    device = resolve_device(device_override or config.training.device)
    seed_everything(config.training.seed)
    step_results: list[PersistentStepResult] = []
    while current.committed_step < target_step:
        step = advance_one_step(
            config,
            bundle_store=bundle_store,
            current=current,
            batch_cursor=current.committed_step,
            device=device,
            parameter_working_set_bytes=parameter_working_set_bytes,
            gradient_working_set_bytes=gradient_working_set_bytes,
            optimizer_working_set_bytes=optimizer_working_set_bytes,
            bundle_failure_injector=bundle_failure_injector,
        )
        step_results.append(step)
        current = bundle_store.current_manifest()

    lineage = _lineage(bundle_store)
    if lineage[-1].committed_step != target_step:
        raise IntegrityError("final lineage step does not match requested target")
    current_state, current_vs_restored, optimizer_steps = _current_bundle_state(
        config,
        bundle_store,
        device,
    )
    resident_state = _resident_reference_state(
        config,
        target_step=target_step,
        device=device,
    )
    bounded_vs_resident = compare_states(resident_state, current_state)
    verification = bundle_store.verify(current.bundle_id)
    result = BoundedTrainingResult(
        schema_version=BOUNDED_TRAINING_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        bundle_store_path=str(destination),
        training_metadata=metadata,
        requested_target_step=target_step,
        started_step=started_step,
        final_step=current.committed_step,
        resumed=resumed,
        initialized_bundle_id=initialized_bundle_id,
        initialization_publication=initialization_publication,
        steps=tuple(step_results),
        lineage=lineage,
        final_bundle_verification=verification,
        final_bounded_vs_resident_state=bounded_vs_resident,
        final_bundle_vs_restored_state=current_vs_restored,
        final_batch_cursor=current.committed_step,
        optimizer_step_values=optimizer_steps,
        total_parameter_logical_bytes_read=sum(
            item.total_parameter_logical_bytes_read for item in step_results
        ),
        total_gradient_logical_bytes_read=sum(
            item.total_gradient_logical_bytes_read for item in step_results
        ),
        total_optimizer_logical_bytes_read=sum(
            item.total_optimizer_logical_bytes_read for item in step_results
        ),
        total_parameter_logical_bytes_written=sum(
            item.total_parameter_logical_bytes_written for item in step_results
        ),
        total_parameter_physical_bytes_written=sum(
            item.total_parameter_physical_bytes_written for item in step_results
        ),
        total_optimizer_logical_bytes_written=sum(
            item.total_optimizer_logical_bytes_written for item in step_results
        ),
        total_optimizer_physical_bytes_written=sum(
            item.total_optimizer_physical_bytes_written for item in step_results
        ),
    )
    if output_path is not None:
        write_json_atomic(output_path, result)
    del current_state, resident_state
    gc.collect()
    synchronize_accelerator(device)
    return result
