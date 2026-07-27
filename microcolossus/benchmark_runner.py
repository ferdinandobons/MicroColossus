"""Backend dispatch and benchmark result serialization."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

from .benchmark_data import (
    _state_artifact_path,
    array_mapping_bytes,
    array_mapping_checksum,
    batch_bytes,
    batch_checksum,
    build_portable_batches,
    build_portable_state,
    portable_parameter_count,
    repository_commit,
    save_array_mapping_atomic,
    summarize_steps,
    system_memory_sample,
)
from .benchmark_types import (
    SCHEMA_VERSION,
    BackendMeasurements,
    BenchmarkResult,
    BenchmarkSettings,
)
from .config import ExperimentConfig
from .telemetry import write_json_atomic


def create_result(
    *,
    config: ExperimentConfig,
    settings: BenchmarkSettings,
    measurements: BackendMeasurements,
    parameter_count: int,
    portable_state_bytes: int,
    portable_batch_bytes: int,
    portable_state_checksum: str,
    batch_checksum_value: str,
    initial_state_released_before_measurement: bool,
    final_state_file: str,
    final_state_checksum: str,
    initial_system_swap_used_bytes: int,
) -> BenchmarkResult:
    tokens_per_step = (
        config.training.micro_batch_size * config.training.sequence_length
    )
    warnings = measurements.warnings
    if not initial_state_released_before_measurement:
        warnings += (
            "The portable NumPy initialization state remained live during measurement.",
        )
    return BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        backend=settings.backend,
        framework=measurements.framework,
        framework_version=measurements.framework_version,
        python_version=sys.version.split()[0],
        numpy_version=str(np.__version__),
        device=measurements.device,
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
        parameter_count=parameter_count,
        portable_state_bytes=portable_state_bytes,
        portable_batch_bytes=portable_batch_bytes,
        initialization_policy="portable-numpy-pcg64-normal-fp32-v1",
        portable_state_checksum=portable_state_checksum,
        batch_policy="portable-numpy-pcg64-int32-v1",
        batch_checksum=batch_checksum_value,
        initial_state_released_before_measurement=(
            initial_state_released_before_measurement
        ),
        final_state_file=final_state_file,
        final_state_checksum=final_state_checksum,
        memory_semantics=measurements.memory_semantics,
        steps=measurements.steps,
        summary=summarize_steps(
            measurements.steps,
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

    initial_system_swap_used_bytes = system_memory_sample()[2]
    total_steps = settings.warmup_steps + settings.measured_steps
    state = build_portable_state(config.model, config.training.seed)
    initial_parameter_count = portable_parameter_count(state)
    initial_state_bytes = array_mapping_bytes(state)
    initial_state_checksum = array_mapping_checksum(state)
    batches = build_portable_batches(config, total_steps)
    batches_bytes = batch_bytes(batches)
    batches_checksum = batch_checksum(batches)

    if settings.backend == "pytorch":
        from .backends.pytorch_resident import run_pytorch_benchmark

        measurements = run_pytorch_benchmark(config, settings, state, batches)
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
        measurements = run_mlx_benchmark(config, settings, state, batches)

    state_released = not state
    final_state = measurements.final_state
    if portable_parameter_count(final_state) != initial_parameter_count:
        raise RuntimeError(
            "final backend state parameter count differs from initialization"
        )
    state_path = _state_artifact_path(output)
    save_array_mapping_atomic(state_path, final_state)
    final_checksum = array_mapping_checksum(final_state)
    final_state.clear()

    result = create_result(
        config=config,
        settings=settings,
        measurements=measurements,
        parameter_count=initial_parameter_count,
        portable_state_bytes=initial_state_bytes,
        portable_batch_bytes=batches_bytes,
        portable_state_checksum=initial_state_checksum,
        batch_checksum_value=batches_checksum,
        initial_state_released_before_measurement=state_released,
        final_state_file=state_path.name,
        final_state_checksum=final_checksum,
        initial_system_swap_used_bytes=initial_system_swap_used_bytes,
    )
    write_json_atomic(output, result.to_dict())
    return result
