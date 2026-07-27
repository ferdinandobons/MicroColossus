"""Configuration models and YAML loading for the initial prototype."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the controlled decoder-only Transformer."""

    vocab_size: int = 256
    max_sequence_length: int = 128
    layers: int = 2
    heads: int = 4
    hidden_size: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0
    norm_epsilon: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.vocab_size, "vocab_size"),
            (self.max_sequence_length, "max_sequence_length"),
            (self.layers, "layers"),
            (self.heads, "heads"),
            (self.hidden_size, "hidden_size"),
            (self.mlp_ratio, "mlp_ratio"),
        ):
            _positive(value, name)
        if self.hidden_size % self.heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        _positive(self.norm_epsilon, "norm_epsilon")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ModelConfig:
        return cls(**dict(values))


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the resident reference training loop."""

    steps: int = 2
    micro_batch_size: int = 2
    sequence_length: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip_norm: float | None = 1.0
    seed: int = 42
    device: str = "auto"
    mode: str = "reference"

    def __post_init__(self) -> None:
        for value, name in (
            (self.steps, "steps"),
            (self.micro_batch_size, "micro_batch_size"),
            (self.sequence_length, "sequence_length"),
        ):
            _positive(value, name)
        _positive(self.learning_rate, "learning_rate")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm is not None:
            _positive(self.gradient_clip_norm, "gradient_clip_norm")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.mode != "reference":
            raise ValueError("only reference mode is implemented")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        return cls(**dict(values))


@dataclass(frozen=True)
class HardwareBudget:
    """Hard budget targets consumed by the static planner."""

    vram_gib: float = 8.0
    process_ram_gib: float = 5.0
    nvme_gib: float = 100.0
    ssd_write_budget_tb: float = 100.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.vram_gib, "vram_gib"),
            (self.process_ram_gib, "process_ram_gib"),
            (self.nvme_gib, "nvme_gib"),
            (self.ssd_write_budget_tb, "ssd_write_budget_tb"),
        ):
            _positive(value, name)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> HardwareBudget:
        return cls(**dict(values))


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration used by the CLI."""

    name: str
    output_dir: str
    model: ModelConfig
    training: TrainingConfig
    hardware: HardwareBudget

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        if self.training.sequence_length > self.model.max_sequence_length:
            raise ValueError("training.sequence_length exceeds model.max_sequence_length")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExperimentConfig:
        name = str(values.get("name", "experiment"))
        output_dir = str(values.get("output_dir", "runs/experiment"))
        model = ModelConfig.from_mapping(_mapping(values.get("model", {}), "model"))
        training = TrainingConfig.from_mapping(
            _mapping(values.get("training", {}), "training")
        )
        hardware = HardwareBudget.from_mapping(
            _mapping(values.get("hardware", {}), "hardware")
        )
        return cls(
            name=name,
            output_dir=output_dir,
            model=model,
            training=training,
            hardware=hardware,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.from_mapping(_mapping(raw, "configuration"))
