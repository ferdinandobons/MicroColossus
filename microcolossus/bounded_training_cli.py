"""Command-line entry point for persistent bounded training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcolossus-bounded-train",
        description=(
            "Advance a persistent MicroColossus bounded-training bundle to a target step."
        ),
    )
    parser.add_argument("--config", required=True, help="path to an experiment YAML file")
    parser.add_argument(
        "--bundle-store",
        required=True,
        help="new or existing persistent step-bundle directory",
    )
    parser.add_argument("--target-step", required=True, type=int)
    parser.add_argument("--output", required=True, help="result JSON path")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="override configured execution device",
    )
    parser.add_argument(
        "--parameter-working-set-mib",
        type=float,
        default=1.0,
        help="maximum logical parameter bytes materialized for one group",
    )
    parser.add_argument(
        "--gradient-working-set-mib",
        type=float,
        default=1.0,
        help="maximum logical gradient bytes materialized for one group",
    )
    parser.add_argument(
        "--optimizer-working-set-mib",
        type=float,
        default=4.0,
        help="maximum logical parameter, gradient, and Adam bytes for one group",
    )
    parser.add_argument(
        "--activation-working-set-mib",
        type=float,
        default=1.0,
        help="maximum retained activation and activation-gradient bytes",
    )
    parser.add_argument(
        "--workspace-working-set-mib",
        type=float,
        default=4.0,
        help="maximum logical local forward/backward activation workspace",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .bounded_training import run_bounded_training
    from .config import load_experiment_config

    mib = 1024**2
    result = run_bounded_training(
        load_experiment_config(args.config),
        bundle_store_path=args.bundle_store,
        target_step=args.target_step,
        output_path=args.output,
        device_override=args.device,
        parameter_working_set_bytes=int(args.parameter_working_set_mib * mib),
        gradient_working_set_bytes=int(args.gradient_working_set_mib * mib),
        optimizer_working_set_bytes=int(args.optimizer_working_set_mib * mib),
        activation_working_set_bytes=int(args.activation_working_set_mib * mib),
        workspace_working_set_bytes=int(args.workspace_working_set_mib * mib),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0
