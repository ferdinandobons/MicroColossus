"""Resident reference training loop and numerical baseline utilities."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .planner import build_static_plan
from .telemetry import (
    JsonlWriter,
    model_checksum,
    peak_vram_bytes,
    process_rss_bytes,
    reset_peak_vram,
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
    peak_vram_bytes: int
    parameter_checksum: str


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch random generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Resolve an explicit or automatic execution device."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
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
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().double().pow(2).sum().cpu()
    return float(torch.sqrt(squared_norm).item())


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
    reset_peak_vram(device)

    started = time.perf_counter()
    output = model(input_ids, targets)
    if output.loss is None:
        raise RuntimeError("the model did not return a training loss")
    output.loss.backward()
    gradient_norm = _gradient_norm(model)
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration_seconds = time.perf_counter() - started

    return StepMetrics(
        step=step,
        loss=float(output.loss.detach().cpu().item()),
        gradient_norm=gradient_norm,
        duration_seconds=duration_seconds,
        process_rss_bytes=process_rss_bytes(),
        peak_vram_bytes=peak_vram_bytes(device),
        parameter_checksum=model_checksum(model),
    )


def run_resident_experiment(
    config: ExperimentConfig,
    *,
    steps_override: int | None = None,
    device_override: str | None = None,
) -> list[StepMetrics]:
    """Run the current resident reference implementation and persist telemetry."""

    training = config.training
    if steps_override is not None:
        training = replace(training, steps=steps_override)
    if device_override is not None:
        training = replace(training, device=device_override)
    config = replace(config, training=training)

    seed_everything(training.seed)
    device = resolve_device(training.device)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(training.seed + 1)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / "steps.jsonl"
    steps_path.unlink(missing_ok=True)
    telemetry_writer = JsonlWriter(steps_path)
    write_json_atomic(output_dir / "resolved-config.json", config.to_dict())
    write_json_atomic(output_dir / "memory-plan.json", build_static_plan(model, config).to_dict())

    metrics: list[StepMetrics] = []
    for step in range(training.steps):
        input_ids, targets = make_synthetic_lm_batch(
            batch_size=training.micro_batch_size,
            sequence_length=training.sequence_length,
            vocab_size=config.model.vocab_size,
            generator=generator,
        )
        step_metrics = run_resident_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            device=device,
            step=step,
            gradient_clip_norm=training.gradient_clip_norm,
        )
        telemetry_writer.append(step_metrics)
        metrics.append(step_metrics)

    summary = {
        "experiment": config.name,
        "device": str(device),
        "steps": len(metrics),
        "final_loss": metrics[-1].loss if metrics else None,
        "parameter_count": model.parameter_count,
        "final_parameter_checksum": metrics[-1].parameter_checksum if metrics else None,
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
