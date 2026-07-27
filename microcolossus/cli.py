"""Command-line interface for the MicroColossus prototype."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict


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
            "MicroColossus storage, bounded execution, resident baselines, "
            "competitive benchmarks, MPS diagnostics, and static planning."
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
    benchmark.add_argument("--backend", choices=("pytorch", "mlx"), required=True)
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

    store_init = subcommands.add_parser(
        "store-init", help="create an empty versioned tensor store"
    )
    store_init.add_argument("--path", required=True, help="tensor store directory")
    store_init.add_argument("--chunk-size-mib", type=int, default=4)
    store_init.add_argument("--max-storage-gib", type=float, default=100.0)
    store_init.add_argument("--max-staging-mib", type=int, default=16)

    store_verify = subcommands.add_parser(
        "store-verify", help="verify manifests, chunks, and tensor checksums"
    )
    store_verify.add_argument("--path", required=True, help="tensor store directory")
    store_verify.add_argument("--manifest", default=None, help="optional manifest ID")

    store_recover = subcommands.add_parser(
        "store-recover", help="recover incomplete transactions without publishing them"
    )
    store_recover.add_argument("--path", required=True, help="tensor store directory")

    storage_step = subcommands.add_parser(
        "storage-step",
        help="compare one resident update with an observable storage-backed update",
    )
    storage_step.add_argument(
        "--config", required=True, help="path to an experiment YAML file"
    )
    storage_step.add_argument("--store", required=True, help="new tensor store directory")
    storage_step.add_argument("--output", required=True, help="result JSON path")
    storage_step.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="override configured execution device",
    )

    bounded_forward = subcommands.add_parser(
        "bounded-forward",
        help="compare resident forward with parameter groups loaded from storage",
    )
    bounded_forward.add_argument(
        "--config", required=True, help="path to an experiment YAML file"
    )
    bounded_forward.add_argument(
        "--store", required=True, help="new parameter tensor store directory"
    )
    bounded_forward.add_argument("--output", required=True, help="result JSON path")
    bounded_forward.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="override configured execution device",
    )
    bounded_forward.add_argument(
        "--parameter-working-set-mib",
        type=float,
        default=1.0,
        help="maximum logical parameter bytes materialized for one execution group",
    )

    bounded_backward = subcommands.add_parser(
        "bounded-backward",
        help="compare resident gradients with group-bounded backward propagation",
    )
    bounded_backward.add_argument(
        "--config", required=True, help="path to an experiment YAML file"
    )
    bounded_backward.add_argument(
        "--parameter-store",
        required=True,
        help="new immutable parameter tensor store directory",
    )
    bounded_backward.add_argument(
        "--gradient-store",
        required=True,
        help="new versioned gradient tensor store directory",
    )
    bounded_backward.add_argument("--output", required=True, help="result JSON path")
    bounded_backward.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="override configured execution device",
    )
    bounded_backward.add_argument(
        "--parameter-working-set-mib",
        type=float,
        default=1.0,
        help="maximum logical parameter bytes materialized for one group",
    )
    bounded_backward.add_argument(
        "--gradient-working-set-mib",
        type=float,
        default=1.0,
        help="maximum logical gradient bytes materialized for one group",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "store-init":
        from .storage import StoreLimits, VersionedTensorStore

        mib = 1024**2
        gib = 1024**3
        store = VersionedTensorStore.create(
            args.path,
            limits=StoreLimits(
                chunk_size_bytes=args.chunk_size_mib * mib,
                max_storage_bytes=int(args.max_storage_gib * gib),
                max_staging_bytes=args.max_staging_mib * mib,
            ),
        )
        print(json.dumps(store.current_manifest().to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "store-verify":
        from .storage import VersionedTensorStore

        report = VersionedTensorStore.open(args.path).verify(args.manifest)
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0

    if args.command == "store-recover":
        from .storage import VersionedTensorStore

        report = VersionedTensorStore.open(args.path).recover()
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0

    from .config import load_experiment_config
    from .environment import collect_environment
    from .model import DecoderOnlyTransformer
    from .planner import build_static_plan
    from .training import format_step_metrics, run_resident_experiment, seed_everything

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

    if args.command == "storage-step":
        from .storage_training import run_observable_storage_step

        result = run_observable_storage_step(
            config,
            store_path=args.store,
            output_path=args.output,
            device_override=args.device,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "bounded-forward":
        from .bounded_forward import run_bounded_forward

        mib = 1024**2
        result = run_bounded_forward(
            config,
            store_path=args.store,
            output_path=args.output,
            device_override=args.device,
            parameter_working_set_bytes=int(args.parameter_working_set_mib * mib),
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "bounded-backward":
        from .bounded_backward import run_bounded_backward

        mib = 1024**2
        result = run_bounded_backward(
            config,
            parameter_store_path=args.parameter_store,
            gradient_store_path=args.gradient_store,
            output_path=args.output,
            device_override=args.device,
            parameter_working_set_bytes=int(args.parameter_working_set_mib * mib),
            gradient_working_set_bytes=int(args.gradient_working_set_mib * mib),
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
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
