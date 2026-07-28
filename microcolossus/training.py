"""Resident reference training loop and numerical baseline utilities."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .data import prepare_data_source
from .model import DecoderOnlyTransformer
from .planner import build_static_plan
from .telemetry import (
    JsonlWriter,
    accelerator_memory_metrics,
    model_checksum,
    process_rss_bytes,
    reset_accelerator_memory,
    synchronize_accelerator,
    write_json_atomic,
)


@dataclass(frozen=True)
class StepMetrics:
    """Measurements produced by one resident optimizer step."""

    step: int
    loss: float
    gradient_norm: float
    duration_seconds: float
    process_rss_bytes: int
    accelerator_memory_measurement: str
    accelerator_allocated_bytes: int
    accelerator_driver_allocated_bytes: int
    accelerator_recommended_max_bytes: int
    parameter_checksum: str


def mps_is_available() -> bool:
    """Return whether the current PyTorch runtime can execute on MPS."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def mps_is_built() -> bool:
    """Return whether the installed PyTorch binary was built with MPS support."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built())


def cuda_is_available() -> bool:
    """Return whether CUDA is available."""

    return bool(torch.cuda.is_available())


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch random generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_is_available():
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """Resolve an explicit or automatic execution device.

    MPS is preferred by ``auto`` because Apple Silicon is the current primary
    development target. CUDA remains supported as a secondary backend.
    """

    if name == "auto":
        if mps_is_available():
            return torch.device("mps")
        if cuda_is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not mps_is_available():
        if not mps_is_built():
            raise RuntimeError("MPS was requested but this PyTorch build has no MPS support")
        raise RuntimeError("MPS was requested but is not available on this machine")
    if name == "cuda" and not cuda_is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def make_synthetic_lm_batch(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Create deterministic next-token data for infrastructure validation."""

    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length + 1),
        generator=generator,
        dtype=torch.long,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def _gradient_norm(model: nn.Module) -> float:
    """Calculate the global gradient norm without creating float64 MPS tensors."""

    squared_norms: list[float] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_sum = parameter.grad.detach().float().pow(2).sum()
            squared_norms.append(float(squared_sum.cpu().item()))
    return math.sqrt(math.fsum(squared_norms))


def run_resident_step(
    *,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    step: int,
    gradient_clip_norm: float | None,
) -> StepMetrics:
    """Execute one full-parameter resident training step."""

    model.train()
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    reset_accelerator_memory(device)

    started = time.perf_counter()
    output = model(input_ids, targets)
    loss = output.loss
    if loss is None:
        raise RuntimeError("the model did not return a training loss")
    loss.backward()
    gradient_norm = _gradient_norm(model)
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    synchronize_accelerator(device)
    duration_seconds = time.perf_counter() - started
    memory = accelerator_memory_metrics(device)

    return StepMetrics(
        step=step,
        loss=float(loss.detach().cpu().item()),
        gradient_norm=gradient_norm,
        duration_seconds=duration_seconds,
        process_rss_bytes=process_rss_bytes(),
        accelerator_memory_measurement=memory.measurement_kind,
        accelerator_allocated_bytes=memory.allocated_bytes,
        accelerator_driver_allocated_bytes=memory.driver_allocated_bytes,
        accelerator_recommended_max_bytes=memory.recommended_max_bytes,
        parameter_checksum=model_checksum(model),
    )


def run_resident_experiment(
    config: ExperimentConfig,
    *,
    steps_override: int | None = None,
    device_override: str | None = None,
) -> list[StepMetrics]:
    """Run the resident reference implementation and persist telemetry."""

    training = config.training
    if steps_override is not None:
        training = replace(training, steps=steps_override)
    if device_override is not None:
        training = replace(training, device=device_override)
    config = replace(config, training=training)

    seed_everything(training.seed)
    data_source = prepare_data_source(config)
    device = resolve_device(training.device)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / "steps.jsonl"
    steps_path.unlink(missing_ok=True)
    telemetry_writer = JsonlWriter(steps_path)
    write_json_atomic(output_dir / "resolved-config.json", config.to_dict())
    write_json_atomic(output_dir / "data-identity.json", data_source.identity)
    plan = build_static_plan(model, config)
    write_json_atomic(output_dir / "memory-plan.json", plan.to_dict())

    metrics: list[StepMetrics] = []
    for step in range(training.steps):
        batch = data_source.training_batch(step)
        step_metrics = run_resident_step(
            model=model,
            optimizer=optimizer,
            input_ids=batch.input_ids,
            targets=batch.targets,
            device=device,
            step=step,
            gradient_clip_norm=training.gradient_clip_norm,
        )
        telemetry_writer.append(step_metrics)
        metrics.append(step_metrics)

    summary = {
        "experiment": config.name,
        "device": str(device),
        "data_source_kind": data_source.identity.source_kind,
        "data_identity_checksum": data_source.identity.identity_checksum,
        "steps": len(metrics),
        "final_loss": metrics[-1].loss if metrics else None,
        "parameter_count": model.parameter_count,
        "final_parameter_checksum": metrics[-1].parameter_checksum if metrics else None,
        "memory_architecture": plan.memory_architecture,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return metrics


def format_step_metrics(metrics: StepMetrics) -> str:
    """Render one telemetry record for terminal output."""

    values = asdict(metrics)
    values["duration_seconds"] = round(metrics.duration_seconds, 6)
    values["loss"] = round(metrics.loss, 6)
    values["gradient_norm"] = round(metrics.gradient_norm, 6)
    return json.dumps(values, sort_keys=True)
