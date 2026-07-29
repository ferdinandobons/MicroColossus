"""Persistent multi-step bounded training with checkpoint and resume."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import torch

from .activation_planner import (
    ActivationMeasurementProfile,
    ActivationPlan,
    ActivationPlanIntegrityError,
    ActivationProfileIntegrityError,
    build_activation_measurement_profile,
    build_activation_plan,
    load_activation_plan,
    load_activation_profile,
    write_activation_plan,
    write_activation_profile,
)
from .bounded_optimizer import _optimizer_records, _store_payloads
from .bounded_training_types import BoundedTrainingResult, PersistentStepResult
from .config import ExperimentConfig
from .data import PreparedDataSource
from .evaluation import ensure_progress_record, load_progress_records
from .model import DecoderOnlyTransformer
from .persistent_step import advance_one_step
from .pruning import assert_pruning_inactive
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
    _initialize_training_root,
    _lineage,
    _load_training_metadata,
    _open_referenced_store,
    _prepare_data_source_for_run,
    _validate_resume_metadata,
)

HYBRID_PROFILE_FILENAME = "HYBRID_ACTIVATION_PROFILE.json"
HYBRID_PLAN_FILENAME = "HYBRID_ACTIVATION_PLAN.json"

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
    data_source: PreparedDataSource,
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
        batch = data_source.training_batch(cursor)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.input_ids.to(device), batch.targets.to(device))
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


def _ensure_current_progress(
    config: ExperimentConfig,
    *,
    bundle_store: StepBundleStore,
    data_source: PreparedDataSource,
    device: torch.device,
) -> None:
    current = bundle_store.current_manifest()
    if current.committed_step == 0:
        batch_cursor = None
        batch_seed = None
        batch_offsets: tuple[int, ...] = ()
    else:
        batch_cursor = current.committed_step - 1
        batch = data_source.training_batch(batch_cursor)
        batch_seed = batch.seed
        batch_offsets = batch.offsets
    ensure_progress_record(
        config,
        bundle_store=bundle_store,
        manifest=current,
        data_source=data_source,
        device=device,
        batch_cursor=batch_cursor,
        batch_seed=batch_seed,
        batch_source_kind=data_source.identity.source_kind,
        batch_offsets=batch_offsets,
        batch_checksum=current.batch_checksum,
        training_loss=None,
        gradient_norm=None,
        clipping_coefficient=None,
    )


def _build_hybrid_activation_artifacts(
    config: ExperimentConfig,
    *,
    device: torch.device,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> tuple[ActivationMeasurementProfile | None, ActivationPlan | None]:
    if config.training.activation_policy != "hybrid":
        return None, None
    profile = build_activation_measurement_profile(
        config,
        backend="pytorch",
        device_identity=str(device),
        dtype="float32",
    )
    plan = build_activation_plan(
        profile,
        activation_budget_bytes=activation_working_set_bytes,
        workspace_budget_bytes=workspace_working_set_bytes,
        max_replay_depth=config.training.activation_anchor_policy.max_replay_depth,
        fixed_interval=config.training.activation_anchor_policy.fixed_interval,
    )
    if not plan.feasible:
        raise ActivationPlanIntegrityError(
            f"hybrid activation plan is infeasible: {plan.rejection_reason}"
        )
    return profile, plan


def _write_hybrid_activation_artifacts(
    destination: Path,
    profile: ActivationMeasurementProfile | None,
    plan: ActivationPlan | None,
) -> None:
    if profile is None or plan is None:
        return
    write_activation_profile(destination / HYBRID_PROFILE_FILENAME, profile)
    write_activation_plan(destination / HYBRID_PLAN_FILENAME, plan)


def _validate_hybrid_activation_artifacts(
    destination: Path,
    profile: ActivationMeasurementProfile | None,
    plan: ActivationPlan | None,
) -> None:
    if profile is None or plan is None:
        return
    profile_path = destination / HYBRID_PROFILE_FILENAME
    plan_path = destination / HYBRID_PLAN_FILENAME
    if not profile_path.exists() or not plan_path.exists():
        raise ResumeConfigurationError("hybrid activation profile or plan is missing")
    try:
        stored_profile = load_activation_profile(profile_path)
        stored_plan = load_activation_plan(plan_path)
        stored_plan.validate(stored_profile)
    except (ActivationPlanIntegrityError, ActivationProfileIntegrityError) as exc:
        raise ResumeConfigurationError("hybrid activation plan is invalid") from exc
    mismatches: list[str] = []
    if stored_profile.profile_checksum != profile.profile_checksum:
        mismatches.append("activation_profile_checksum")
    if stored_plan.plan_checksum != plan.plan_checksum:
        mismatches.append("activation_plan_checksum")
    if stored_plan.activation_budget_bytes != plan.activation_budget_bytes:
        mismatches.append("activation_budget_bytes")
    if stored_plan.workspace_budget_bytes != plan.workspace_budget_bytes:
        mismatches.append("workspace_budget_bytes")
    if stored_plan.max_replay_depth != plan.max_replay_depth:
        mismatches.append("max_replay_depth")
    if stored_plan.selected_anchor_group_names != plan.selected_anchor_group_names:
        mismatches.append("activation_anchor_group_names")
    if mismatches:
        raise ResumeConfigurationError(
            "hybrid activation plan does not match the requested configuration: "
            + ", ".join(mismatches)
        )


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
    activation_working_set_bytes: int = 1024**2,
    workspace_working_set_bytes: int = 4 * 1024**2,
    bundle_failure_injector: BundleFailureInjector | None = None,
) -> BoundedTrainingResult:
    """Advance a persistent bounded training root to ``target_step``."""

    if target_step < 0:
        raise ValueError("target_step cannot be negative")
    for value, name in (
        (parameter_working_set_bytes, "parameter_working_set_bytes"),
        (gradient_working_set_bytes, "gradient_working_set_bytes"),
        (optimizer_working_set_bytes, "optimizer_working_set_bytes"),
        (activation_working_set_bytes, "activation_working_set_bytes"),
        (workspace_working_set_bytes, "workspace_working_set_bytes"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    data_source = _prepare_data_source_for_run(config)
    destination = Path(bundle_store_path)
    initialization_publication: BundlePublicationTelemetry | None = None
    device = resolve_device(device_override or config.training.device)
    activation_profile, activation_plan = _build_hybrid_activation_artifacts(
        config,
        device=device,
        activation_working_set_bytes=activation_working_set_bytes,
        workspace_working_set_bytes=workspace_working_set_bytes,
    )
    if destination.exists():
        assert_pruning_inactive(destination)
        bundle_store = StepBundleStore.open(destination)
        metadata = _load_training_metadata(destination)
        _validate_resume_metadata(
            config,
            metadata,
            data_source,
            activation_profile=activation_profile,
            activation_plan=activation_plan,
        )
        _validate_hybrid_activation_artifacts(
            destination,
            activation_profile,
            activation_plan,
        )
        current = bundle_store.current_manifest()
        resumed = True
        initialized_bundle_id = _lineage(bundle_store)[0].bundle_id
    else:
        bundle_store, current, initialization_publication, metadata = (
            _initialize_training_root(
                config,
                destination,
                data_source,
                activation_profile=activation_profile,
                activation_plan=activation_plan,
            )
        )
        _write_hybrid_activation_artifacts(destination, activation_profile, activation_plan)
        resumed = False
        initialized_bundle_id = current.bundle_id
    bundle_store.verify(current.bundle_id)
    started_step = current.committed_step
    if target_step < started_step:
        raise ValueError(
            f"target_step {target_step} is behind current committed step {started_step}"
        )

    seed_everything(config.training.seed)
    _ensure_current_progress(
        config,
        bundle_store=bundle_store,
        data_source=data_source,
        device=device,
    )
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
            activation_working_set_bytes=activation_working_set_bytes,
            workspace_working_set_bytes=workspace_working_set_bytes,
            activation_profile=activation_profile,
            activation_plan=activation_plan,
            bundle_failure_injector=bundle_failure_injector,
        )
        step_results.append(step)
        current = bundle_store.current_manifest()
        batch = data_source.training_batch(step.batch_cursor)
        ensure_progress_record(
            config,
            bundle_store=bundle_store,
            manifest=current,
            data_source=data_source,
            device=device,
            batch_cursor=step.batch_cursor,
            batch_seed=step.batch_seed,
            batch_source_kind=batch.source_kind,
            batch_offsets=batch.offsets,
            batch_checksum=step.batch_checksum,
            training_loss=step.bounded_loss,
            gradient_norm=step.bounded_gradient_norm,
            clipping_coefficient=step.clipping_coefficient,
        )

    lineage = _lineage(bundle_store)
    if lineage[-1].committed_step != target_step:
        raise IntegrityError("final lineage step does not match requested target")
    progress = load_progress_records(destination)
    if tuple(item.step for item in progress) != tuple(item.committed_step for item in lineage):
        raise IntegrityError("progress steps do not match root bundle lineage")
    bundle_ids_match = all(
        record.bundle_id == item.bundle_id
        for record, item in zip(progress, lineage, strict=True)
    )
    if not bundle_ids_match:
        raise IntegrityError("progress bundle IDs do not match root bundle lineage")
    current_state, current_vs_restored, optimizer_steps = _current_bundle_state(
        config,
        bundle_store,
        device,
    )
    resident_state = _resident_reference_state(
        config,
        target_step=target_step,
        device=device,
        data_source=data_source,
    )
    bounded_vs_resident = compare_states(resident_state, current_state)
    verification = bundle_store.verify(current.bundle_id)
    result = BoundedTrainingResult(
        schema_version=BOUNDED_TRAINING_SCHEMA_VERSION,
        experiment=config.name,
        device=str(device),
        bundle_store_path=str(destination),
        training_metadata=metadata,
        data_identity=data_source.identity,
        requested_target_step=target_step,
        started_step=started_step,
        final_step=current.committed_step,
        resumed=resumed,
        initialized_bundle_id=initialized_bundle_id,
        initialization_publication=initialization_publication,
        activation_policy=config.training.activation_policy,
        activation_working_set_budget_bytes=activation_working_set_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_bytes,
        steps=tuple(step_results),
        lineage=lineage,
        progress_records=progress,
        metrics_directory=str(destination / "metrics"),
        final_bundle_verification=verification,
        final_bounded_vs_resident_state=bounded_vs_resident,
        final_bundle_vs_restored_state=current_vs_restored,
        final_batch_cursor=current.committed_step,
        optimizer_step_values=optimizer_steps,
        maximum_retained_activation_bytes=max(
            (item.maximum_retained_activation_bytes for item in step_results),
            default=0,
        ),
        maximum_workspace_bytes=max(
            (item.maximum_workspace_bytes for item in step_results),
            default=0,
        ),
        maximum_retained_forward_boundary_bytes=max(
            (item.retained_forward_boundary_bytes for item in step_results),
            default=0,
        ),
        total_prefix_replayed_groups=sum(
            item.total_prefix_replayed_groups for item in step_results
        ),
        total_prefix_recomputation_seconds=sum(
            item.total_prefix_recomputation_seconds for item in step_results
        ),
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
