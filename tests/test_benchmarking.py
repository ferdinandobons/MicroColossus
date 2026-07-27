from microcolossus.benchmarking import (
    BenchmarkSettings,
    array_mapping_checksum,
    build_portable_batches,
    build_portable_state,
    compare_benchmarks,
    portable_parameter_count,
    run_benchmark,
)
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.model import DecoderOnlyTransformer


def _experiment_config(tmp_path):
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


def test_portable_state_matches_unique_model_parameters(tmp_path):
    config = _experiment_config(tmp_path)
    state = build_portable_state(config.model, config.training.seed)
    model = DecoderOnlyTransformer(config.model)
    assert portable_parameter_count(state) == model.parameter_count
    assert array_mapping_checksum(state) == array_mapping_checksum(state)


def test_batches_are_reproducible(tmp_path):
    config = _experiment_config(tmp_path)
    first = build_portable_batches(config, 3)
    second = build_portable_batches(config, 3)
    for (first_x, first_y), (second_x, second_y) in zip(first, second, strict=True):
        assert (first_x == second_x).all()
        assert (first_y == second_y).all()


def test_pytorch_cpu_benchmark_and_comparison(tmp_path):
    config = _experiment_config(tmp_path)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    settings = BenchmarkSettings(backend="pytorch", warmup_steps=1, measured_steps=2)
    left = run_benchmark(config, settings, left_path)
    right = run_benchmark(config, settings, right_path)
    assert left.summary.tokens_per_second > 0
    assert len(left.steps) == 3
    comparison = compare_benchmarks(left_path, right_path, tmp_path / "compare.json")
    assert comparison.equivalent_portable_state
    assert comparison.equivalent_batches
    assert comparison.equivalent_benchmark_schedule


def test_checkpointing_cpu_benchmark(tmp_path):
    config = _experiment_config(tmp_path)
    settings = BenchmarkSettings(
        backend="pytorch",
        warmup_steps=0,
        measured_steps=1,
        activation_checkpointing=True,
    )
    result = run_benchmark(config, settings, tmp_path / "checkpoint.json")
    assert result.activation_checkpointing
