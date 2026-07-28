"""Command-line interface for the activation-recomputation reference path."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .activation_recompute import run_activation_recompute_validation
from .config import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m microcolossus.activation_recompute_cli",
        description=(
            "Validate zero-boundary-retention activation recomputation against "
            "resident full-parameter gradients."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-store", required=True)
    parser.add_argument("--oracle-gradient-store", required=True)
    parser.add_argument("--gradient-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--parameter-working-set-mib", type=float, default=1.0)
    parser.add_argument("--gradient-working-set-mib", type=float, default=1.0)
    parser.add_argument("--activation-working-set-mib", type=float, default=1.0)
    parser.add_argument("--workspace-working-set-mib", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mib = 1024**2
    result = run_activation_recompute_validation(
        load_experiment_config(args.config),
        parameter_store_path=args.parameter_store,
        oracle_gradient_store_path=args.oracle_gradient_store,
        gradient_store_path=args.gradient_store,
        output_path=args.output,
        device_override=args.device,
        parameter_working_set_bytes=int(args.parameter_working_set_mib * mib),
        gradient_working_set_bytes=int(args.gradient_working_set_mib * mib),
        activation_working_set_bytes=int(args.activation_working_set_mib * mib),
        workspace_working_set_bytes=int(args.workspace_working_set_mib * mib),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
