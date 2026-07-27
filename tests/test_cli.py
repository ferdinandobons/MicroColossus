import json

from microcolossus.cli import main


def test_plan_command_emits_json(capsys) -> None:
    result = main(["plan", "--config", "examples/tiny-resident.yaml"])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["estimate_kind"] == "static-model-plus-heuristics-v1"
    assert output["parameter_count"] > 0


def test_doctor_command_reports_mps_fields(capsys) -> None:
    result = main(["doctor"])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert "mps_built" in output
    assert "mps_available" in output
    assert "system_memory_bytes" in output
