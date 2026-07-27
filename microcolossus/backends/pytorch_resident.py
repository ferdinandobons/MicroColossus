"""Competitive resident benchmark for the PyTorch backend."""

from __future__ import annotations

import gc
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

from ..benchmark_data import system_memory_sample
from ..benchmark_types import (
    BackendMeasurements,
    BenchmarkSettings,
    BenchmarkStep,
)
from ..config import ExperimentConfig
from ..model import DecoderOnlyTransformer
from ..telemetry import (
    accelerator_memory_metrics,
    process_rss_bytes,
    reset_accelerator_memory,
    synchronize_accelerator,
)
from ..training import resolve_device, seed_everything


def _framework_version() -> str:
    try:
        return version("torch")
    except PackageNotFoundError:
        return str(torch.__version__)


def _load_portable_state(
    model: DecoderOnlyTransformer, state: dict[str, np.ndarray]
) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    unexpected = sorted(set(parameters) - set(state))
    if missing or unexpected:
        raise ValueError(
            f"portable state mismatch: missing_in_model={missing}, "
            f"missing_in_state={unexpected}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            source = torch.from_numpy(value).to(
                device=parameter.device, dtype=parameter.dtype
            )
            parameter.copy_(source)


def _export_state(model: DecoderOnlyTransformer) -> dict[str, np.ndarray]:
    return {
        name: parameter.detach().cpu().float().contiguous().numpy().copy()
        for name, parameter in model.named_parameters()
    }


def run_pytorch_benchmark(
    config: ExperimentConfig,
    settings: BenchmarkSettings,
    state: dict[str, np.ndarray],
    batches: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> BackendMeasurements:
    """Run synchronized training steps without checksums in the timed region."""

    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    model = DecoderOnlyTransformer(config.model).to(device)
    _load_portable_state(model, state)
    state.clear()
    gc.collect()

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    records: list[BenchmarkStep] = []
    for index, (input_array, target_array) in enumerate(batches):
        input_ids = torch.from_numpy(input_array).to(
            device=device, dtype=torch.long
        )
        targets = torch.from_numpy(target_array).to(
            device=device, dtype=torch.long
        )
        synchronize_accelerator(device)
        reset_accelerator_memory(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids,
            targets,
            activation_checkpointing=settings.activation_checkpointing,
        )
        loss = output.loss
        if loss is None:
            raise RuntimeError("the PyTorch model did not return a loss")
        loss.backward()
        if config.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
        optimizer.step()
        synchronize_accelerator(device)
        duration = time.perf_counter() - started
        memory = accelerator_memory_metrics(device)
        available, memory_percent, swap_used = system_memory_sample()
        phase = "warmup" if index < settings.warmup_steps else "measured"
        phase_index = index if phase == "warmup" else index - settings.warmup_steps
        records.append(
            BenchmarkStep(
                phase=phase,
                index=phase_index,
                loss=float(loss.detach().cpu().item()),
                duration_seconds=duration,
                process_rss_bytes=process_rss_bytes(),
                memory_measurement_kind=memory.measurement_kind,
                framework_memory_bytes=memory.allocated_bytes,
                driver_memory_bytes=memory.driver_allocated_bytes,
                cache_memory_bytes=0,
                system_available_memory_bytes=available,
                system_memory_percent=memory_percent,
                system_swap_used_bytes=swap_used,
            )
        )

    warnings = [
        "Input conversion is performed before the timed region.",
        "Per-step model checksums are disabled because they force full CPU readback.",
        "The portable NumPy initialization arrays are released before warm-up.",
        "Final parameters are exported after the timed region.",
    ]
    if device.type == "mps":
        warnings.append(
            "PyTorch MPS reports current tensor and driver allocations, not a "
            "resettable per-step physical-memory peak."
        )
    return BackendMeasurements(
        framework="PyTorch",
        framework_version=_framework_version(),
        device=str(device),
        steps=tuple(records),
        memory_semantics=(
            "PyTorch allocator counters. On MPS, current tensor and Metal driver "
            "values overlap with process and unified-memory accounting."
        ),
        warnings=tuple(warnings),
        final_state=_export_state(model),
    )
