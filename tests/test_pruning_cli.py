from __future__ import annotations

import json
from pathlib import Path

from microcolossus.bounded_training import run_bounded_training
from microcolossus.config import load_experiment_config
from microcolossus.pruning_cli import main
from microcolossus.step_bundle import StepBundleStore


def test_pruning_cli_plan_apply_and_repeat(tmp_path: Path, capsys) -> None:
    config_path = Path("examples/micro-storage.yaml")
    config = load_experiment_config(config_path)
    root = tmp_path / "training"
    run_bounded_training(
        config,
        bundle_store_path=root,
        target_step=3,
        device_override="cpu",
    )
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    repeated_path = tmp_path / "report-repeated.json"

    assert (
        main(
            [
                "plan",
                "--config",
                str(config_path),
                "--bundle-store",
                str(root),
                "--output",
                str(plan_path),
                "--keep-previous",
                "0",
                "--milestone-interval",
                "0",
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["current_step"] == 3
    assert plan["retained_steps"] == [3]
    assert plan["selected_bytes"] > 0

    assert (
        main(
            [
                "apply",
                "--config",
                str(config_path),
                "--bundle-store",
                str(root),
                "--plan",
                str(plan_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["current_pointer_unchanged"] is True
    assert report["idempotent"] is False
    assert report["cumulative_reclaimed_bytes"] == plan["selected_bytes"]
    StepBundleStore.open(root).verify()

    assert (
        main(
            [
                "apply",
                "--config",
                str(config_path),
                "--bundle-store",
                str(root),
                "--plan",
                str(plan_path),
                "--output",
                str(repeated_path),
            ]
        )
        == 0
    )
    repeated = json.loads(repeated_path.read_text(encoding="utf-8"))
    assert repeated["idempotent"] is True
    assert repeated["newly_reclaimed_bytes"] == 0
    assert "plan_checksum" in capsys.readouterr().out
