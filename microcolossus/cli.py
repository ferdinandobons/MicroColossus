"""Command-line interface for the MicroColossus prototype."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .config import load_experiment_config
from .environment import collect_environment
from .model import DecoderOnlyTransformer
from .planner import build_static_plan
from .training import format_step_metrics, run_resident_experiment, seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcolossus",
        description="MicroColossus resident baseline, MPS diagnostics, and static planner.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(collect_environment().to_dict(), indent=2, sort_keys=True))
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

    raise AssertionError(f"unhandled command: {args.command}")
