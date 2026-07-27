from pathlib import Path

import pytest

from microcolossus.config import HardwareBudget, TrainingConfig, load_experiment_config


def test_example_configuration_loads() -> None:
    config = load_experiment_config(Path("examples/tiny-resident.yaml"))
    assert config.name == "tiny-resident"
    assert config.model.hidden_size == 128
    assert config.training.mode == "reference"
    assert config.hardware.accelerator_memory_gib == 8.0


def test_mps_configuration_is_unified_memory() -> None:
    config = load_experiment_config(Path("examples/tiny-mps.yaml"))
    assert config.training.device == "mps"
    assert config.hardware.memory_architecture == "unified"
    assert config.hardware.system_memory_gib == 8.0


def test_mps_is_an_accepted_device() -> None:
    assert TrainingConfig(device="mps").device == "mps"


def test_legacy_vram_budget_is_accepted() -> None:
    budget = HardwareBudget.from_mapping({"vram_gib": 4.0})
    assert budget.accelerator_memory_gib == 4.0


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
