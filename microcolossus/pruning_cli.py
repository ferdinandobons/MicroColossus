"""Command-line interface for deterministic checkpoint pruning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcolossus-prune",
        description=(
            "Plan or explicitly apply safe pruning to a persistent MicroColossus "
            "training root."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="create a deterministic dry-run plan")
    plan.add_argument("--config", required=True, help="experiment YAML path")
    plan.add_argument("--bundle-store", required=True, help="training root")
    plan.add_argument("--output", required=True, help="plan JSON path")
    plan.add_argument(
        "--keep-previous",
        type=int,
        default=None,
        help="override retention.keep_previous",
    )
    plan.add_argument(
        "--milestone-interval",
        type=int,
        default=None,
        help="override retention.milestone_interval; zero disables milestones",
    )

    apply = subparsers.add_parser("apply", help="apply one checksummed pruning plan")
    apply.add_argument("--config", required=True, help="experiment YAML path")
    apply.add_argument("--bundle-store", required=True, help="training root")
    apply.add_argument("--plan", required=True, help="checksummed plan JSON path")
    apply.add_argument("--output", required=True, help="report JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .config import load_experiment_config
    from .pruning import (
        apply_pruning_plan,
        build_pruning_plan,
        load_pruning_plan,
        write_pruning_plan,
        write_pruning_report,
    )

    config = load_experiment_config(args.config)
    if args.command == "plan":
        result = build_pruning_plan(
            config,
            bundle_store_path=args.bundle_store,
            keep_previous=args.keep_previous,
            milestone_interval=args.milestone_interval,
        )
        write_pruning_plan(args.output, result)
    else:
        result = apply_pruning_plan(
            config,
            bundle_store_path=args.bundle_store,
            plan=load_pruning_plan(args.plan),
        )
        write_pruning_report(args.output, result)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0
