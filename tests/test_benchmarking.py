from pathlib import Path

import psutil

from microcolossus.benchmarking import (
    BenchmarkSettings,
    array_mapping_checksum,
    build_portable_batches,
    build_portable_state,
    compare_benchmarks,
    load_array_mapping,
    portable_parameter_count,
    run_benchmark,
    system_memory_sample,
)
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.model import DecoderOnlyTransformer


def _experiment_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="bench",
        output_dir=str(tmp_path / "run"),
        model=ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=1,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
        ),
        training=TrainingConfig(
            steps=2,
            micro_batch_size=1,
            sequence_length=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            seed=3,
            device="cpu",
        ),
        hardware=HardwareBudget(),
    )


def test_portable_state_matches_unique_model_parameters(tmp_path: Path) -> None:
    config = _experiment_config(tmp_path)
    state = build_portable_state(config.model, config.training.seed)
    model = DecoderOnlyTransformer(config.model)
    assert portable_parameter_count(state) == model.parameter_count
    assert array_mapping_checksum(state) == array_mapping_checksum(state)


def test_batches_are_reproducible(tmp_path: Path) -> None:
    config = _experiment_config(tmp_path)
    first = build_portable_batches(config, 3)
    second = build_portable_batches(config, 3)
    for (first_x, first_y), (second_x, second_y) in zip(
        first, second, strict=True
    ):
        assert (first_x == second_x).all()
        assert (first_y == second_y).all()


def test_system_memory_sample_tolerates_unavailable_swap(monkeypatch) -> None:
    def fail_swap() -> None:
        raise OSError("swap telemetry unavailable")

    monkeypatch.setattr(psutil, "swap_memory", fail_swap)
    available_bytes, memory_percent, swap_used_bytes = system_memory_sample()
    assert available_bytes >= 0
    assert memory_percent >= 0.0
    assert swap_used_bytes == 0


def test_pytorch_benchmark_writes_state_and_releases_initial_state(
    tmp_path: Path,
) -> None:
    config = _experiment_config(tmp_path)
    output = tmp_path / "left.json"
    result = run_benchmark(
        config,
        BenchmarkSettings(
            backend="pytorch", warmup_steps=1, measured_steps=2
        ),
        output,
    )
    state_path = tmp_path / result.final_state_file
    state = load_array_mapping(state_path)
    assert state_path.exists()
    assert array_mapping_checksum(state) == result.final_state_checksum
    assert result.initial_state_released_before_measurement
    assert result.portable_state_bytes > 0
    assert result.portable_batch_bytes > 0


def test_identical_pytorch_runs_have_zero_tensor_difference(
    tmp_path: Path,
) -> None:
    config = _experiment_config(tmp_path)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    settings = BenchmarkSettings(
        backend="pytorch", warmup_steps=1, measured_steps=2
    )
    run_benchmark(config, settings, left_path)
    run_benchmark(config, settings, right_path)
    comparison = compare_benchmarks(
        left_path, right_path, tmp_path / "compare.json"
    )
    assert comparison.equivalent_portable_state
    assert comparison.equivalent_batches
    assert comparison.final_state_tensor_names_equal
    assert comparison.final_state_checksums_equal
    assert comparison.final_state_all_values_finite
    assert comparison.final_state_maximum_absolute_difference == 0.0
    assert comparison.final_state_mean_absolute_difference == 0.0
    assert comparison.final_state_maximum_relative_difference == 0.0
    assert comparison.tensor_differences


def test_checkpointing_cpu_benchmark(tmp_path: Path) -> None:
    config = _experiment_config(tmp_path)
    result = run_benchmark(
        config,
        BenchmarkSettings(
            backend="pytorch",
            warmup_steps=0,
            measured_steps=1,
            activation_checkpointing=True,
        ),
        tmp_path / "checkpoint.json",
    )
    assert result.activation_checkpointing
