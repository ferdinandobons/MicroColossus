"""Configuration models and YAML loading for MicroColossus experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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


def _resolve_optional_path(value: Any, base_dir: Path | None) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return str(path.resolve())


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
class ActivationAnchorPolicyConfig:
    """Configuration for a measured hybrid activation-anchor plan."""

    kind: str = "measured_budget_v1"
    fixed_interval: int = 2
    max_replay_depth: int | None = None

    def __post_init__(self) -> None:
        if self.kind != "measured_budget_v1":
            raise ValueError("only measured_budget_v1 activation anchors are implemented")
        _positive(self.fixed_interval, "activation_anchor_policy.fixed_interval")
        if self.max_replay_depth is not None:
            if self.max_replay_depth < 0:
                raise ValueError(
                    "activation_anchor_policy.max_replay_depth cannot be negative"
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ActivationAnchorPolicyConfig:
        return cls(**dict(values))


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for resident and bounded full-parameter training."""

    steps: int = 2
    micro_batch_size: int = 2
    sequence_length: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip_norm: float | None = 1.0
    seed: int = 42
    device: str = "auto"
    mode: str = "reference"
    activation_policy: str = "retain_all"
    activation_anchor_policy: ActivationAnchorPolicyConfig = field(
        default_factory=ActivationAnchorPolicyConfig
    )

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
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.mode != "reference":
            raise ValueError("only reference mode is implemented")
        if self.activation_policy not in {"retain_all", "recompute", "hybrid"}:
            raise ValueError(
                "training.activation_policy must be one of: retain_all, recompute, hybrid"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        normalized = dict(values)
        anchor_policy = normalized.get("activation_anchor_policy", {})
        if isinstance(anchor_policy, ActivationAnchorPolicyConfig):
            normalized["activation_anchor_policy"] = anchor_policy
        else:
            normalized["activation_anchor_policy"] = (
                ActivationAnchorPolicyConfig.from_mapping(
                    _mapping(anchor_policy, "training.activation_anchor_policy")
                )
            )
        return cls(**normalized)


@dataclass(frozen=True)
class DataConfig:
    """Deterministic training-data frontend configuration."""

    kind: str = "synthetic"
    train_path: str | None = None
    validation_path: str | None = None
    validation_fraction: float = 0.1
    tokenizer: str = "utf8-bytes-v1"
    sampler: str = "random-window-v1"

    def __post_init__(self) -> None:
        if self.kind not in {"synthetic", "utf8_text"}:
            raise ValueError("data.kind must be one of: synthetic, utf8_text")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("data.validation_fraction must be in (0, 1)")
        if self.tokenizer != "utf8-bytes-v1":
            raise ValueError("only utf8-bytes-v1 tokenizer is implemented")
        if self.sampler != "random-window-v1":
            raise ValueError("only random-window-v1 sampler is implemented")
        if self.kind == "synthetic":
            if self.train_path is not None or self.validation_path is not None:
                raise ValueError("synthetic data cannot define corpus paths")
        elif self.train_path is None:
            raise ValueError("utf8_text data requires train_path")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> DataConfig:
        normalized = dict(values)
        normalized["train_path"] = _resolve_optional_path(
            normalized.get("train_path"), base_dir
        )
        normalized["validation_path"] = _resolve_optional_path(
            normalized.get("validation_path"), base_dir
        )
        return cls(**normalized)


@dataclass(frozen=True)
class EvaluationConfig:
    """Validation-loss and deterministic sample-generation configuration."""

    enabled: bool = False
    interval_steps: int = 1
    validation_batches: int = 2
    sample_tokens: int = 0
    sample_prompt: str = ""
    generation: str = "greedy-v1"

    def __post_init__(self) -> None:
        _positive(self.interval_steps, "evaluation.interval_steps")
        _positive(self.validation_batches, "evaluation.validation_batches")
        if self.sample_tokens < 0:
            raise ValueError("evaluation.sample_tokens cannot be negative")
        if self.generation != "greedy-v1":
            raise ValueError("only greedy-v1 generation is implemented")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> EvaluationConfig:
        return cls(**dict(values))


@dataclass(frozen=True)
class RetentionConfig:
    """Explicit checkpoint-retention policy used by pruning commands."""

    keep_previous: int = 2
    milestone_interval: int = 0

    def __post_init__(self) -> None:
        if self.keep_previous < 0:
            raise ValueError("retention.keep_previous cannot be negative")
        if self.milestone_interval < 0:
            raise ValueError("retention.milestone_interval cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RetentionConfig:
        return cls(**dict(values))


@dataclass(frozen=True)
class HardwareBudget:
    """Logical memory and storage budgets consumed by the static planner.

    On Apple Silicon, accelerator and process-memory budgets share one physical
    unified-memory pool. They are logical guardrails and must not be added.
    """

    accelerator_memory_gib: float = 8.0
    process_ram_gib: float = 5.0
    nvme_gib: float = 100.0
    ssd_write_budget_tb: float = 100.0
    memory_architecture: str = "auto"
    system_memory_gib: float | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.accelerator_memory_gib, "accelerator_memory_gib"),
            (self.process_ram_gib, "process_ram_gib"),
            (self.nvme_gib, "nvme_gib"),
            (self.ssd_write_budget_tb, "ssd_write_budget_tb"),
        ):
            _positive(value, name)
        if self.memory_architecture not in {"auto", "discrete", "unified"}:
            raise ValueError("memory_architecture must be one of: auto, discrete, unified")
        if self.system_memory_gib is not None:
            _positive(self.system_memory_gib, "system_memory_gib")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> HardwareBudget:
        normalized = dict(values)
        legacy_vram = normalized.pop("vram_gib", None)
        if legacy_vram is not None:
            if "accelerator_memory_gib" in normalized:
                raise ValueError(
                    "hardware cannot define both vram_gib and accelerator_memory_gib"
                )
            normalized["accelerator_memory_gib"] = legacy_vram
        return cls(**normalized)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration used by the CLI."""

    name: str
    output_dir: str
    model: ModelConfig
    training: TrainingConfig
    hardware: HardwareBudget
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        if self.training.sequence_length > self.model.max_sequence_length:
            raise ValueError("training.sequence_length exceeds model.max_sequence_length")
        if self.data.kind == "utf8_text" and self.model.vocab_size != 256:
            raise ValueError("utf8_text requires model.vocab_size == 256")
        if self.evaluation.enabled and self.data.kind != "utf8_text":
            raise ValueError("evaluation is currently implemented only for utf8_text data")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> ExperimentConfig:
        name = str(values.get("name", "experiment"))
        output_dir = str(values.get("output_dir", "runs/experiment"))
        model = ModelConfig.from_mapping(_mapping(values.get("model", {}), "model"))
        training = TrainingConfig.from_mapping(
            _mapping(values.get("training", {}), "training")
        )
        hardware = HardwareBudget.from_mapping(
            _mapping(values.get("hardware", {}), "hardware")
        )
        data = DataConfig.from_mapping(
            _mapping(values.get("data", {}), "data"),
            base_dir=base_dir,
        )
        evaluation = EvaluationConfig.from_mapping(
            _mapping(values.get("evaluation", {}), "evaluation")
        )
        retention = RetentionConfig.from_mapping(
            _mapping(values.get("retention", {}), "retention")
        )
        return cls(
            name=name,
            output_dir=output_dir,
            model=model,
            training=training,
            hardware=hardware,
            data=data,
            evaluation=evaluation,
            retention=retention,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.from_mapping(
        _mapping(raw, "configuration"),
        base_dir=config_path.parent,
    )
