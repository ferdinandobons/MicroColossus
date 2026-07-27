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
    assert first_output["schema_version"] == "microcolossus.bounded-training.v1"
    assert first_output["final_step"] == 1
    assert second == 0
    assert second_output["resumed"]
    assert second_output["started_step"] == 1
    assert second_output["final_step"] == 2
    assert second_output["final_bounded_vs_resident_state"]["exact_bytes"]
