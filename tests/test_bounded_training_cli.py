from __future__ import annotations

import json
from pathlib import Path

from microcolossus.bounded_training_cli import main


def test_bounded_training_cli_initializes_and_resumes(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "bundle"
    first = main(
        [
            "--config",
            "examples/micro-storage.yaml",
            "--bundle-store",
            str(bundle),
            "--target-step",
            "1",
            "--output",
            str(tmp_path / "step-1.json"),
            "--device",
            "cpu",
        ]
    )
    first_output = json.loads(capsys.readouterr().out)
    second = main(
        [
            "--config",
            "examples/micro-storage.yaml",
            "--bundle-store",
            str(bundle),
            "--target-step",
            "2",
            "--output",
            str(tmp_path / "step-2.json"),
            "--device",
            "cpu",
        ]
    )
    second_output = json.loads(capsys.readouterr().out)

    assert first == 0
    assert first_output["schema_version"] == "microcolossus.bounded-training.v3"
    assert first_output["activation_policy"] == "retain_all"
    assert first_output["final_step"] == 1
    assert second == 0
    assert second_output["resumed"]
    assert second_output["started_step"] == 1
    assert second_output["final_step"] == 2
    assert second_output["final_bounded_vs_resident_state"]["exact_bytes"]


def test_bounded_training_cli_runs_persistent_recompute(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "--config",
            "examples/real-text-micro-recompute.yaml",
            "--bundle-store",
            str(tmp_path / "recompute-bundle"),
            "--target-step",
            "1",
            "--output",
            str(tmp_path / "recompute-step-1.json"),
            "--device",
            "cpu",
            "--parameter-working-set-mib",
            "1",
            "--gradient-working-set-mib",
            "1",
            "--optimizer-working-set-mib",
            "4",
            "--activation-working-set-mib",
            "1",
            "--workspace-working-set-mib",
            "4",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["schema_version"] == "microcolossus.bounded-training.v3"
    assert output["activation_policy"] == "recompute"
    assert output["final_step"] == 1
    assert output["maximum_retained_forward_boundary_bytes"] == 0
    assert output["total_prefix_replayed_groups"] > 0
    assert output["final_bundle_vs_restored_state"]["exact_bytes"]
