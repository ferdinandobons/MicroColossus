from __future__ import annotations

import json
from pathlib import Path

from microcolossus.cli import main


def test_bounded_step_command_emits_json(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "bounded-step",
            "--config",
            "examples/micro-storage.yaml",
            "--bundle-store",
            str(tmp_path / "bundle"),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cpu",
            "--parameter-working-set-mib",
            "1",
            "--gradient-working-set-mib",
            "1",
            "--optimizer-working-set-mib",
            "4",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["schema_version"] == "microcolossus.bounded-optimizer.v1"
    assert output["resident_vs_candidate_state"]["exact_bytes"]
    assert output["candidate_vs_restored_state"]["exact_bytes"]
    assert output["final_bundle_is_authoritative"]
