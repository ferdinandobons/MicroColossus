"""Public competitive benchmark API."""

from .benchmark_compare import compare_benchmarks, load_benchmark_result
from .benchmark_data import (
    array_mapping_bytes,
    array_mapping_checksum,
    batch_bytes,
    batch_checksum,
    build_portable_batches,
    build_portable_state,
    load_array_mapping,
    portable_parameter_count,
    save_array_mapping_atomic,
    system_memory_sample,
)
from .benchmark_runner import create_result, run_benchmark
from .benchmark_types import (
    BackendMeasurements,
    BenchmarkComparison,
    BenchmarkResult,
    BenchmarkSettings,
    BenchmarkStep,
    BenchmarkSummary,
    TensorDifference,
)

__all__ = [
    "BackendMeasurements",
    "BenchmarkComparison",
    "BenchmarkResult",
    "BenchmarkSettings",
    "BenchmarkStep",
    "BenchmarkSummary",
    "TensorDifference",
    "array_mapping_bytes",
    "array_mapping_checksum",
    "batch_bytes",
    "batch_checksum",
    "build_portable_batches",
    "build_portable_state",
    "compare_benchmarks",
    "create_result",
    "load_array_mapping",
    "load_benchmark_result",
    "portable_parameter_count",
    "run_benchmark",
    "save_array_mapping_atomic",
    "system_memory_sample",
]
