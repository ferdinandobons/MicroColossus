import json

from microcolossus.cli import main


def test_plan_command_emits_json(capsys) -> None:
    result = main(["plan", "--config", "examples/tiny-resident.yaml"])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["estimate_kind"] == "static-model-plus-heuristics-v0"
    assert output["parameter_count"] > 0
