"""Backend-neutral data structures for competitive resident benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .config import ExperimentConfig, ModelConfig
from .telemetry import write_json_atomic

SCHEMA_VERSION = "microcolossus.benchmark.v1"


@dataclass(frozen=True)
class BenchmarkSettings:
    """Controls warm-up and measured resident training steps."""

    backend: str
    warmup_steps: int = 2
    measured_steps: int = 10
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"pytorch", "mlx"}:
            raise ValueError("backend must be one of: pytorch, mlx")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.measured_steps <= 0:
            raise ValueError("measured_steps must be greater than zero")
        if self.backend == "mlx" and self.activation_checkpointing:
            raise ValueError(
                "MLX activation checkpointing is not implemented in this harness"
            )


@dataclass(frozen=True)
class BenchmarkStep:
    """One synchronized warm-up or measured optimizer step."""

    phase: str
    index: int
    loss: float
    duration_seconds: float
    process_rss_bytes: int
    memory_measurement_kind: str
    framework_memory_bytes: int
    driver_memory_bytes: int
    cache_memory_bytes: int
    system_available_memory_bytes: int
    system_memory_percent: float
    system_swap_used_bytes: int


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregated values derived only from the measured phase."""

    first_step_seconds: float
    mean_step_seconds: float
    median_step_seconds: float
    p95_step_seconds: float
    min_step_seconds: float
    max_step_seconds: float
    tokens_per_second: float
    initial_measured_loss: float
    final_measured_loss: float
    max_process_rss_bytes: int
    max_framework_memory_bytes: int
    max_driver_memory_bytes: int
    max_cache_memory_bytes: int
    min_system_available_memory_bytes: int
    max_system_memory_percent: float
    initial_system_swap_used_bytes: int
    final_system_swap_used_bytes: int
    system_swap_delta_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Machine-readable output for one framework and backend."""

    schema_version: str
    backend: str
    framework: str
    framework_version: str
    python_version: str
    numpy_version: str
    device: str
    repository_commit: str | None
    platform: str
    machine: str
    system_memory_bytes: int
    model: dict[str, Any]
    training: dict[str, Any]
    warmup_steps: int
    measured_steps: int
    activation_checkpointing: bool
    tokens_per_step: int
    parameter_count: int
    initialization_policy: str
    portable_state_checksum: str
    batch_policy: str
    batch_checksum: str
    memory_semantics: str
    steps: tuple[BenchmarkStep, ...]
    summary: BenchmarkSummary
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison that keeps framework memory semantics separate."""

    schema_version: str
    left_backend: str
    right_backend: str
    equivalent_model_config: bool
    equivalent_training_config: bool
    equivalent_benchmark_schedule: bool
    equivalent_activation_checkpointing: bool
    equivalent_portable_state: bool
    equivalent_batches: bool
    loss_trajectories_same_length: bool
    max_absolute_loss_difference: float | None
    mean_absolute_loss_difference: float | None
    final_absolute_loss_difference: float | None
    left_tokens_per_second: float
    right_tokens_per_second: float
    right_over_left_throughput: float
    left_mean_step_seconds: float
    right_mean_step_seconds: float
    left_memory_measurement_kind: str
    right_memory_measurement_kind: str
    left_max_framework_memory_bytes: int
    right_max_framework_memory_bytes: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normal(
    generator: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    return generator.normal(
        loc=0.0, scale=0.02, size=shape
    ).astype(np.float32)


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
    state["final_norm.weight"] = np.ones(
        hidden_size, dtype=np.float32
    )
    state["final_norm.bias"] = np.zeros(
        hidden_size, dtype=np.float32
    )
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
    if any(
        not math.isfinite(value) or value <= 0 for value in durations
    ):
        raise ValueError(
            "all measured durations must be finite and positive"
        )
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
        max_process_rss_bytes=max(
            step.process_rss_bytes for step in measured
        ),
        max_framework_memory_bytes=max(
            step.framework_memory_bytes for step in measured
        ),
        max_driver_memory_bytes=max(
            step.driver_memory_bytes for step in measured
        ),
        max_cache_memory_bytes=max(
            step.cache_memory_bytes for step in measured
        ),
        min_system_available_memory_bytes=min(
            step.system_available_memory_bytes for step in measured
        ),
        max_system_memory_percent=max(
            step.system_memory_percent for step in measured
        ),
        initial_system_swap_used_bytes=initial_system_swap_used_bytes,
        final_system_swap_used_bytes=steps[-1].system_swap_used_bytes,
        system_swap_delta_bytes=(
            steps[-1].system_swap_used_bytes
            - initial_system_swap_used_bytes
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


def create_result(
    *,
    config: ExperimentConfig,
    settings: BenchmarkSettings,
    framework: str,
    framework_version: str,
    device: str,
    state: dict[str, np.ndarray],
    batches: tuple[tuple[np.ndarray, np.ndarray], ...],
    steps: tuple[BenchmarkStep, ...],
    memory_semantics: str,
    initial_system_swap_used_bytes: int,
    warnings: tuple[str, ...] = (),
) -> BenchmarkResult:
    tokens_per_step = (
        config.training.micro_batch_size
        * config.training.sequence_length
    )
    return BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        backend=settings.backend,
        framework=framework,
        framework_version=framework_version,
        python_version=sys.version.split()[0],
        numpy_version=str(np.__version__),
        device=device,
        repository_commit=repository_commit(),
        platform=platform.platform(),
        machine=platform.machine(),
        system_memory_bytes=int(psutil.virtual_memory().total),
        model=asdict(config.model),
        training=asdict(config.training),
        warmup_steps=settings.warmup_steps,
        measured_steps=settings.measured_steps,
        activation_checkpointing=settings.activation_checkpointing,
        tokens_per_step=tokens_per_step,
        parameter_count=portable_parameter_count(state),
        initialization_policy="portable-numpy-pcg64-normal-fp32-v1",
        portable_state_checksum=array_mapping_checksum(state),
        batch_policy="portable-numpy-pcg64-int32-v1",
        batch_checksum=batch_checksum(batches),
        memory_semantics=memory_semantics,
        steps=steps,
        summary=summarize_steps(
            steps,
            tokens_per_step,
            initial_system_swap_used_bytes,
        ),
        warnings=warnings,
    )


def run_benchmark(
    config: ExperimentConfig,
    settings: BenchmarkSettings,
    output: str | Path,
) -> BenchmarkResult:
    """Dispatch to a backend imported only when selected."""

    total_steps = settings.warmup_steps + settings.measured_steps
    state = build_portable_state(
        config.model, config.training.seed
    )
    batches = build_portable_batches(config, total_steps)
    if settings.backend == "pytorch":
        from .backends.pytorch_resident import run_pytorch_benchmark

        result = run_pytorch_benchmark(
            config, settings, state, batches
        )
    else:
        try:
            from .backends.mlx_resident import run_mlx_benchmark
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.startswith("mlx"):
                raise RuntimeError(
                    "MLX is not installed. Install benchmark dependencies "
                    'with python -m pip install -e ".[benchmark]" on '
                    "native Apple Silicon."
                ) from exc
            raise
        result = run_mlx_benchmark(
            config, settings, state, batches
        )
    write_json_atomic(output, result.to_dict())
    return result


def load_benchmark_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(
            f"{path} is not a {SCHEMA_VERSION} result"
        )
    return value


def compare_benchmarks(
    left_path: str | Path,
    right_path: str | Path,
    output: str | Path,
) -> BenchmarkComparison:
    left = load_benchmark_result(left_path)
    right = load_benchmark_result(right_path)
    left_summary = left["summary"]
    right_summary = right["summary"]
    left_throughput = float(left_summary["tokens_per_second"])
    right_throughput = float(right_summary["tokens_per_second"])
    if left_throughput <= 0:
        raise ValueError(
            "left benchmark throughput must be positive"
        )
    left_losses = [
        float(step["loss"])
        for step in left["steps"]
        if step["phase"] == "measured"
    ]
    right_losses = [
        float(step["loss"])
        for step in right["steps"]
        if step["phase"] == "measured"
    ]
    same_loss_length = len(left_losses) == len(right_losses)
    loss_differences = (
        [
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_losses,
                right_losses,
                strict=True,
            )
        ]
        if same_loss_length
        else []
    )
    left_kind = str(
        left["steps"][-1]["memory_measurement_kind"]
    )
    right_kind = str(
        right["steps"][-1]["memory_measurement_kind"]
    )
    warnings = (
        "Framework memory counters can describe different allocator scopes "
        "and are not physical-memory equivalents.",
        "Use macOS memory pressure, swap, and process footprint as "
        "external cross-checks.",
    )
    comparison = BenchmarkComparison(
        schema_version="microcolossus.benchmark-comparison.v1",
        left_backend=str(left["backend"]),
        right_backend=str(right["backend"]),
        equivalent_model_config=left["model"] == right["model"],
        equivalent_training_config=(
            left["training"] == right["training"]
        ),
        equivalent_benchmark_schedule=(
            left["warmup_steps"] == right["warmup_steps"]
            and left["measured_steps"] == right["measured_steps"]
        ),
        equivalent_activation_checkpointing=(
            left["activation_checkpointing"]
            == right["activation_checkpointing"]
        ),
        equivalent_portable_state=(
            left["portable_state_checksum"]
            == right["portable_state_checksum"]
        ),
        equivalent_batches=(
            left["batch_checksum"] == right["batch_checksum"]
        ),
        loss_trajectories_same_length=same_loss_length,
        max_absolute_loss_difference=(
            max(loss_differences) if loss_differences else None
        ),
        mean_absolute_loss_difference=(
            statistics.fmean(loss_differences)
            if loss_differences
            else None
        ),
        final_absolute_loss_difference=(
            loss_differences[-1] if loss_differences else None
        ),
        left_tokens_per_second=left_throughput,
        right_tokens_per_second=right_throughput,
        right_over_left_throughput=(
            right_throughput / left_throughput
        ),
        left_mean_step_seconds=float(
            left_summary["mean_step_seconds"]
        ),
        right_mean_step_seconds=float(
            right_summary["mean_step_seconds"]
        ),
        left_memory_measurement_kind=left_kind,
        right_memory_measurement_kind=right_kind,
        left_max_framework_memory_bytes=int(
            left_summary["max_framework_memory_bytes"]
        ),
        right_max_framework_memory_bytes=int(
            right_summary["max_framework_memory_bytes"]
        ),
        warnings=warnings,
    )
    write_json_atomic(output, comparison.to_dict())
    return comparison
