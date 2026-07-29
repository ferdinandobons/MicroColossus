"""Command-line interface for measured activation-anchor planning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .activation_planner import (
    build_activation_measurement_profile,
    build_activation_plan,
    write_activation_plan,
    write_activation_profile,
)
from .config import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcolossus-activation-plan",
        description="Build a checksummed M6C activation measurement profile and plan.",
    )
    parser.add_argument("--config", required=True, help="path to an experiment YAML file")
    parser.add_argument("--profile-output", required=True, help="profile JSON output")
    parser.add_argument("--plan-output", required=True, help="plan JSON output")
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--device-identity", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--activation-working-set-mib", type=float, default=1.0)
    parser.add_argument("--workspace-working-set-mib", type=float, default=4.0)
    parser.add_argument("--fixed-interval", type=int, default=2)
    parser.add_argument("--max-replay-depth", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    mib = 1024**2
    profile = build_activation_measurement_profile(
        config,
        backend=args.backend,
        device_identity=args.device_identity,
        dtype=args.dtype,
    )
    plan = build_activation_plan(
        profile,
        activation_budget_bytes=int(args.activation_working_set_mib * mib),
        workspace_budget_bytes=int(args.workspace_working_set_mib * mib),
        max_replay_depth=args.max_replay_depth,
        fixed_interval=args.fixed_interval,
    )
    write_activation_profile(args.profile_output, profile)
    write_activation_plan(args.plan_output, plan)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
