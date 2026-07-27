"""Portable data and measurement helpers for competitive benchmarks."""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import subprocess
from pathlib import Path

import numpy as np
import psutil

from .benchmark_types import BenchmarkStep, BenchmarkSummary
from .config import ExperimentConfig, ModelConfig


def _normal(
    generator: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    return generator.normal(loc=0.0, scale=0.02, size=shape).astype(np.float32)


def build_portable_state(
    config: ModelConfig, seed: int
) -> dict[str, np.ndarray]:
    """Create one framework-neutral FP32 parameter state."""

    generator = np.random.default_rng(seed)
    hidden_size = config.hidden_size
    intermediate_size = hidden_size * config.mlp_ratio
    state: dict[str, np.ndarray] = {
        "token_embedding.weight": _normal(
            generator, (config.vocab_size, hidden_size)
        ),
        "position_embedding.weight": _normal(
            generator, (config.max_sequence_length, hidden_size)
        ),
    }
    for index in range(config.layers):
        prefix = f"blocks.{index}"
        state[f"{prefix}.attention_norm.weight"] = np.ones(
            hidden_size, dtype=np.float32
        )
        state[f"{prefix}.attention_norm.bias"] = np.zeros(
            hidden_size, dtype=np.float32
        )
        state[f"{prefix}.attention.qkv.weight"] = _normal(
            generator, (3 * hidden_size, hidden_size)
        )
        state[f"{prefix}.attention.output.weight"] = _normal(
            generator, (hidden_size, hidden_size)
        )
        state[f"{prefix}.mlp_norm.weight"] = np.ones(
            hidden_size, dtype=np.float32
        )
        state[f"{prefix}.mlp_norm.bias"] = np.zeros(
            hidden_size, dtype=np.float32
        )
        state[f"{prefix}.mlp.input.weight"] = _normal(
            generator, (intermediate_size, hidden_size)
        )
        state[f"{prefix}.mlp.output.weight"] = _normal(
            generator, (hidden_size, intermediate_size)
        )
    state["final_norm.weight"] = np.ones(hidden_size, dtype=np.float32)
    state["final_norm.bias"] = np.zeros(hidden_size, dtype=np.float32)
    if not config.tie_embeddings:
        state["lm_head.weight"] = _normal(
            generator, (config.vocab_size, hidden_size)
        )
    return state


def array_mapping_checksum(values: dict[str, np.ndarray]) -> str:
    """Hash names, shapes, data types, and logical bytes."""

    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def array_mapping_bytes(values: dict[str, np.ndarray]) -> int:
    return sum(int(value.nbytes) for value in values.values())


def build_portable_batches(
    config: ExperimentConfig, total_steps: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Generate identical next-token batches for every framework."""

    if total_steps <= 0:
        raise ValueError("total_steps must be greater than zero")
    training = config.training
    shape = (
        total_steps,
        training.micro_batch_size,
        training.sequence_length + 1,
    )
    generator = np.random.default_rng(training.seed + 1)
    tokens = generator.integers(
        0,
        config.model.vocab_size,
        size=shape,
        dtype=np.int32,
    )
    return tuple(
        (
            np.ascontiguousarray(step_tokens[:, :-1]),
            np.ascontiguousarray(step_tokens[:, 1:]),
        )
        for step_tokens in tokens
    )


def batch_checksum(
    batches: tuple[tuple[np.ndarray, np.ndarray], ...]
) -> str:
    digest = hashlib.sha256()
    for input_ids, targets in batches:
        digest.update(input_ids.tobytes(order="C"))
        digest.update(targets.tobytes(order="C"))
    return digest.hexdigest()


def batch_bytes(
    batches: tuple[tuple[np.ndarray, np.ndarray], ...]
) -> int:
    return sum(
        int(input_ids.nbytes + targets.nbytes)
        for input_ids, targets in batches
    )


def portable_parameter_count(state: dict[str, np.ndarray]) -> int:
    return sum(int(value.size) for value in state.values())


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def system_memory_sample() -> tuple[int, float, int]:
    """Return available memory, memory percentage, and swap usage."""

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return int(memory.available), float(memory.percent), int(swap.used)


def summarize_steps(
    steps: tuple[BenchmarkStep, ...],
    tokens_per_step: int,
    initial_system_swap_used_bytes: int,
) -> BenchmarkSummary:
    measured = [step for step in steps if step.phase == "measured"]
    if not measured:
        raise ValueError("at least one measured step is required")
    durations = [step.duration_seconds for step in measured]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("all measured durations must be finite and positive")
    losses = [step.loss for step in measured]
    if any(not math.isfinite(value) for value in losses):
        raise ValueError("all measured losses must be finite")
    mean_seconds = statistics.fmean(durations)
    return BenchmarkSummary(
        first_step_seconds=steps[0].duration_seconds,
        mean_step_seconds=mean_seconds,
        median_step_seconds=statistics.median(durations),
        p95_step_seconds=_percentile(durations, 0.95),
        min_step_seconds=min(durations),
        max_step_seconds=max(durations),
        tokens_per_second=tokens_per_step / mean_seconds,
        initial_measured_loss=losses[0],
        final_measured_loss=losses[-1],
        max_process_rss_bytes=max(step.process_rss_bytes for step in measured),
        max_framework_memory_bytes=max(
            step.framework_memory_bytes for step in measured
        ),
        max_driver_memory_bytes=max(
            step.driver_memory_bytes for step in measured
        ),
        max_cache_memory_bytes=max(step.cache_memory_bytes for step in measured),
        min_system_available_memory_bytes=min(
            step.system_available_memory_bytes for step in measured
        ),
        max_system_memory_percent=max(
            step.system_memory_percent for step in measured
        ),
        initial_system_swap_used_bytes=initial_system_swap_used_bytes,
        final_system_swap_used_bytes=steps[-1].system_swap_used_bytes,
        system_swap_delta_bytes=(
            steps[-1].system_swap_used_bytes - initial_system_swap_used_bytes
        ),
    )


def repository_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    value = completed.stdout.strip()
    return value or None


def _state_artifact_path(output: str | Path) -> Path:
    destination = Path(output)
    return destination.with_name(destination.stem + ".state.npz")


def save_array_mapping_atomic(
    path: str | Path, values: dict[str, np.ndarray]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, destination)


def load_array_mapping(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
        }
