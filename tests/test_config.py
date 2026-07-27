from pathlib import Path

import pytest

from microcolossus.config import load_experiment_config


def test_example_configuration_loads() -> None:
    config = load_experiment_config(Path("examples/tiny-resident.yaml"))
    assert config.name == "tiny-resident"
    assert config.model.hidden_size == 128
    assert config.training.mode == "reference"


def test_sequence_length_must_fit_model(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
name: invalid
output_dir: runs/invalid
model:
  max_sequence_length: 8
training:
  sequence_length: 16
hardware: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds"):
        load_experiment_config(path)
