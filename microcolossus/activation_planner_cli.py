"""Command-line interface for activation profile calibration and plan construction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcolossus-activation-plan",
        description="Calibrate activation costs and build deterministic hybrid anchor plans.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser(
        "profile",
        help="build a measured profile from completed retain-all and recompute runs",
    )
    profile.add_argument("--config", required=True)
    profile.add_argument("--retain-result", required=True)
    profile.add_argument("--recompute-result", required=True)
    profile.add_argument("--device-identity", required=True)
    profile.add_argument("--output", required=True)

    build = commands.add_parser("build", help="build a checksummed anchor plan")
    build.add_argument("--config", required=True)
    build.add_argument("--profile")
    build.add_argument(
        "--kind",
        choices=("measured_budget_v1", "logical_budget_v1", "fixed_interval_v1"),
        default="measured_budget_v1",
    )
    build.add_argument("--activation-working-set-mib", type=float, required=True)
    build.add_argument("--workspace-working-set-mib", type=float, required=True)
    build.add_argument("--max-replay-groups", type=int)
    build.add_argument("--fixed-interval", type=int, default=2)
    build.add_argument("--device-identity", default="logical-estimate-v1")
    build.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .activation_planner import (
        build_activation_plan,
        build_logical_activation_profile,
        build_measured_activation_profile,
        load_activation_profile,
        write_activation_plan,
        write_activation_profile,
    )
    from .config import load_experiment_config

    config = load_experiment_config(args.config)
    if args.command == "profile":
        result = build_measured_activation_profile(
            config,
            retain_result_path=args.retain_result,
            recompute_result_path=args.recompute_result,
            device_identity=args.device_identity,
        )
        write_activation_profile(args.output, result)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    profile = (
        load_activation_profile(args.profile)
        if args.profile is not None
        else build_logical_activation_profile(
            config,
            device_identity=args.device_identity,
        )
    )
    mib = 1024**2
    result = build_activation_plan(
        profile,
        policy_kind=args.kind,
        activation_working_set_budget_bytes=int(
            args.activation_working_set_mib * mib
        ),
        workspace_working_set_budget_bytes=int(args.workspace_working_set_mib * mib),
        max_replay_groups=args.max_replay_groups,
        fixed_interval=args.fixed_interval,
    )
    write_activation_plan(args.output, result)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.feasible else 2
