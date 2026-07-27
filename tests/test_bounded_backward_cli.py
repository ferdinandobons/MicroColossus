from __future__ import annotations

import json
from pathlib import Path

from microcolossus.cli import main


def test_bounded_backward_command_emits_json(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "bounded-backward",
            "--config",
            "examples/micro-storage.yaml",
            "--parameter-store",
            str(tmp_path / "parameters"),
            "--gradient-store",
            str(tmp_path / "gradients"),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cpu",
            "--parameter-working-set-mib",
            "1",
            "--gradient-working-set-mib",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["schema_version"] == "microcolossus.bounded-backward.v1"
    assert output["resident_vs_bounded_gradients"]["exact_bytes"]
    assert output["parameter_manifest_unchanged"]
