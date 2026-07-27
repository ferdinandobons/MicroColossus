"""Command-line interface for the MicroColossus prototype."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .config import load_experiment_config
from .environment import collect_environment
from .model import DecoderOnlyTransformer
from .planner import build_static_plan
from .training import format_step_metrics, run_resident_experiment, seed_everything


def _benchmark_import_error(exc: ModuleNotFoundError) -> RuntimeError:
    dependency = exc.name or "benchmark dependency"
    return RuntimeError(
        f"Missing {dependency}. Install benchmark dependencies with "
        'python -m pip install -e ".[benchmark]".'
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcolossus",
        description=(
            "MicroColossus resident baselines, competitive benchmarks, MPS diagnostics, "
            "and static planning."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="report CPU, CUDA, and Apple MPS availability")

    plan = subcommands.add_parser("plan", help="estimate memory requirements")
    plan.add_argument("--config", required=True, help="path to an experiment YAML file")

    train = subcommands.add_parser("train", help="run the resident reference baseline")
    train.add_argument("--config", required=True, help="path to an experiment YAML file")
    train.add_argument("--steps", type=int, default=None, help="override configured steps")
    train.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="override configured execution device",
    )

    benchmark = subcommands.add_parser(
        "benchmark", help="run a synchronized competitive resident benchmark"
    )
    benchmark.add_argument(
        "--config", required=True, help="path to an experiment YAML file"
    )
    benchmark.add_argument(
        "--backend", choices=("pytorch", "mlx"), required=True
    )
    benchmark.add_argument("--warmup-steps", type=int, default=2)
    benchmark.add_argument("--steps", type=int, default=10, help="measured steps")
    benchmark.add_argument("--output", required=True, help="benchmark JSON output")
    benchmark.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="enable PyTorch activation checkpointing",
    )

    compare = subcommands.add_parser(
        "compare-benchmarks", help="compare two benchmark JSON results"
    )
    compare.add_argument("--left", required=True, help="first benchmark JSON")
    compare.add_argument("--right", required=True, help="second benchmark JSON")
    compare.add_argument("--output", required=True, help="comparison JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(collect_environment().to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "compare-benchmarks":
        try:
            from .benchmarking import compare_benchmarks
        except ModuleNotFoundError as exc:
            raise _benchmark_import_error(exc) from exc

        comparison = compare_benchmarks(args.left, args.right, args.output)
        print(json.dumps(comparison.to_dict(), indent=2, sort_keys=True))
        return 0

    config = load_experiment_config(args.config)
    if args.command == "plan":
        seed_everything(config.training.seed)
        model = DecoderOnlyTransformer(config.model)
        print(json.dumps(build_static_plan(model, config).to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "train":
        metrics = run_resident_experiment(
            config,
            steps_override=args.steps,
            device_override=args.device,
        )
        for item in metrics:
            print(format_step_metrics(item))
        return 0

    if args.command == "benchmark":
        try:
            from .benchmarking import BenchmarkSettings, run_benchmark
        except ModuleNotFoundError as exc:
            raise _benchmark_import_error(exc) from exc

        settings = BenchmarkSettings(
            backend=args.backend,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            activation_checkpointing=args.activation_checkpointing,
        )
        result = run_benchmark(config, settings, args.output)
        print(json.dumps(asdict(result.summary), indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")
