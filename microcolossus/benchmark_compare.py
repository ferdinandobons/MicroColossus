"""Numerical and performance comparison for benchmark artifacts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_data import array_mapping_checksum, load_array_mapping
from .benchmark_types import (
    COMPARISON_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BenchmarkComparison,
    TensorDifference,
)
from .telemetry import write_json_atomic


def load_benchmark_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} is not a {SCHEMA_VERSION} result")
    return value


def _load_result_state(
    result_path: str | Path, result: dict[str, Any]
) -> dict[str, np.ndarray]:
    state_path = Path(str(result["final_state_file"]))
    if not state_path.is_absolute():
        state_path = Path(result_path).parent / state_path
    state = load_array_mapping(state_path)
    expected = str(result["final_state_checksum"])
    actual = array_mapping_checksum(state)
    if actual != expected:
        raise ValueError(
            f"final state checksum mismatch for {result_path}: {actual} != {expected}"
        )
    return state


def _tensor_differences(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> tuple[
    tuple[TensorDifference, ...],
    bool,
    float,
    float,
    float,
    str,
    str,
]:
    differences: list[TensorDifference] = []
    total_absolute = 0.0
    total_values = 0
    maximum_absolute = -1.0
    maximum_relative = -1.0
    worst_absolute = ""
    worst_relative = ""
    all_finite = True
    for name in sorted(left):
        left_value = left[name]
        right_value = right[name]
        if left_value.shape != right_value.shape:
            raise ValueError(f"shape mismatch for final tensor {name}")
        left64 = left_value.astype(np.float64, copy=False)
        right64 = right_value.astype(np.float64, copy=False)
        finite = bool(np.isfinite(left64).all() and np.isfinite(right64).all())
        all_finite = all_finite and finite
        absolute = np.abs(left64 - right64)
        denominator = np.maximum(
            np.maximum(np.abs(left64), np.abs(right64)), 1e-12
        )
        relative = absolute / denominator
        max_absolute = float(absolute.max(initial=0.0))
        mean_absolute = float(absolute.mean()) if absolute.size else 0.0
        max_relative = float(relative.max(initial=0.0))
        total_absolute += float(absolute.sum())
        total_values += int(absolute.size)
        if max_absolute > maximum_absolute:
            maximum_absolute = max_absolute
            worst_absolute = name
        if max_relative > maximum_relative:
            maximum_relative = max_relative
            worst_relative = name
        differences.append(
            TensorDifference(
                name=name,
                shape=tuple(int(value) for value in left_value.shape),
                maximum_absolute_difference=max_absolute,
                mean_absolute_difference=mean_absolute,
                maximum_relative_difference=max_relative,
                all_values_finite=finite,
            )
        )
    global_mean = total_absolute / total_values if total_values else 0.0
    return (
        tuple(differences),
        all_finite,
        max(maximum_absolute, 0.0),
        global_mean,
        max(maximum_relative, 0.0),
        worst_absolute,
        worst_relative,
    )


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
        raise ValueError("left benchmark throughput must be positive")

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
                left_losses, right_losses, strict=True
            )
        ]
        if same_loss_length
        else []
    )

    left_state = _load_result_state(left_path, left)
    right_state = _load_result_state(right_path, right)
    same_names = set(left_state) == set(right_state)
    tensor_differences: tuple[TensorDifference, ...] = ()
    state_all_finite: bool | None = None
    state_max_abs: float | None = None
    state_mean_abs: float | None = None
    state_max_rel: float | None = None
    worst_abs: str | None = None
    worst_rel: str | None = None
    if same_names:
        (
            tensor_differences,
            state_all_finite,
            state_max_abs,
            state_mean_abs,
            state_max_rel,
            worst_abs_value,
            worst_rel_value,
        ) = _tensor_differences(left_state, right_state)
        worst_abs = worst_abs_value or None
        worst_rel = worst_rel_value or None
    left_state.clear()
    right_state.clear()

    left_kind = str(left["steps"][-1]["memory_measurement_kind"])
    right_kind = str(right["steps"][-1]["memory_measurement_kind"])
    comparison = BenchmarkComparison(
        schema_version=COMPARISON_SCHEMA_VERSION,
        left_backend=str(left["backend"]),
        right_backend=str(right["backend"]),
        equivalent_model_config=left["model"] == right["model"],
        equivalent_training_config=left["training"] == right["training"],
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
        equivalent_batches=left["batch_checksum"] == right["batch_checksum"],
        loss_trajectories_same_length=same_loss_length,
        max_absolute_loss_difference=(
            max(loss_differences) if loss_differences else None
        ),
        mean_absolute_loss_difference=(
            statistics.fmean(loss_differences) if loss_differences else None
        ),
        final_absolute_loss_difference=(
            loss_differences[-1] if loss_differences else None
        ),
        final_state_tensor_names_equal=same_names,
        final_state_checksums_equal=(
            left["final_state_checksum"] == right["final_state_checksum"]
        ),
        final_state_all_values_finite=state_all_finite,
        final_state_maximum_absolute_difference=state_max_abs,
        final_state_mean_absolute_difference=state_mean_abs,
        final_state_maximum_relative_difference=state_max_rel,
        final_state_worst_absolute_tensor=worst_abs,
        final_state_worst_relative_tensor=worst_rel,
        tensor_differences=tensor_differences,
        left_tokens_per_second=left_throughput,
        right_tokens_per_second=right_throughput,
        right_over_left_throughput=right_throughput / left_throughput,
        left_mean_step_seconds=float(left_summary["mean_step_seconds"]),
        right_mean_step_seconds=float(right_summary["mean_step_seconds"]),
        left_memory_measurement_kind=left_kind,
        right_memory_measurement_kind=right_kind,
        left_max_framework_memory_bytes=int(
            left_summary["max_framework_memory_bytes"]
        ),
        right_max_framework_memory_bytes=int(
            right_summary["max_framework_memory_bytes"]
        ),
        warnings=(
            "Framework memory counters can describe different allocator scopes "
            "and are not physical-memory equivalents.",
            "Use macOS memory pressure, swap, and process footprint as external "
            "cross-checks.",
            "Maximum relative parameter differences near zero can be large even "
            "when absolute differences are small.",
        ),
    )
    write_json_atomic(output, comparison.to_dict())
    return comparison
