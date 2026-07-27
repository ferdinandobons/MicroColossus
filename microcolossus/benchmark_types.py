"""Schemas for competitive benchmark results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = "microcolossus.benchmark.v2"
COMPARISON_SCHEMA_VERSION = "microcolossus.benchmark-comparison.v2"


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
class BackendMeasurements:
    """Backend output produced outside the serialization layer."""

    framework: str
    framework_version: str
    device: str
    steps: tuple[BenchmarkStep, ...]
    memory_semantics: str
    warnings: tuple[str, ...]
    final_state: dict[str, np.ndarray]


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
    portable_state_bytes: int
    portable_batch_bytes: int
    initialization_policy: str
    portable_state_checksum: str
    batch_policy: str
    batch_checksum: str
    initial_state_released_before_measurement: bool
    final_state_file: str
    final_state_checksum: str
    memory_semantics: str
    steps: tuple[BenchmarkStep, ...]
    summary: BenchmarkSummary
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TensorDifference:
    """Numerical distance between one final parameter tensor."""

    name: str
    shape: tuple[int, ...]
    maximum_absolute_difference: float
    mean_absolute_difference: float
    maximum_relative_difference: float
    all_values_finite: bool


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
    final_state_tensor_names_equal: bool
    final_state_checksums_equal: bool
    final_state_all_values_finite: bool | None
    final_state_maximum_absolute_difference: float | None
    final_state_mean_absolute_difference: float | None
    final_state_maximum_relative_difference: float | None
    final_state_worst_absolute_tensor: str | None
    final_state_worst_relative_tensor: str | None
    tensor_differences: tuple[TensorDifference, ...]
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
