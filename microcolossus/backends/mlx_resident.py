"""Competitive resident benchmark for Apple's MLX framework."""

from __future__ import annotations

import gc
import math
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from ..benchmark_data import system_memory_sample
from ..benchmark_types import (
    BackendMeasurements,
    BenchmarkSettings,
    BenchmarkStep,
)
from ..config import ExperimentConfig, ModelConfig
from ..telemetry import process_rss_bytes


class MLXCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_size = config.hidden_size // config.heads
        self.qkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.output = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def __call__(self, hidden_states: Any) -> Any:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        query, key, value = mx.split(self.qkv(hidden_states), 3, axis=-1)

        def split_heads(value_to_split: Any) -> Any:
            value_to_split = mx.reshape(
                value_to_split,
                (batch_size, sequence_length, self.heads, self.head_size),
            )
            return mx.transpose(value_to_split, (0, 2, 1, 3))

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = (
            query @ mx.transpose(key, (0, 1, 3, 2))
        ) / math.sqrt(self.head_size)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            sequence_length, dtype=scores.dtype
        )
        probabilities = mx.softmax(scores + mask, axis=-1)
        probabilities = self.attention_dropout(probabilities)
        context = probabilities @ value
        context = mx.transpose(context, (0, 2, 1, 3))
        context = mx.reshape(
            context, (batch_size, sequence_length, hidden_size)
        )
        return self.residual_dropout(self.output(context))


class MLXFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        intermediate_size = config.hidden_size * config.mlp_ratio
        self.input = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.output = nn.Linear(
            intermediate_size, config.hidden_size, bias=False
        )
        self.activation = nn.GELU(approx="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, hidden_states: Any) -> Any:
        return self.dropout(
            self.output(self.activation(self.input(hidden_states)))
        )


class MLXTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_epsilon
        )
        self.attention = MLXCausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_epsilon
        )
        self.mlp = MLXFeedForward(config)

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class MLXDecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = [
            MLXTransformerBlock(config) for _ in range(config.layers)
        ]
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_epsilon
        )
        self.lm_head = (
            None
            if config.tie_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

    def __call__(self, input_ids: Any) -> Any:
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds model.max_sequence_length")
        positions = mx.arange(sequence_length)
        hidden_states = self.token_embedding(
            input_ids
        ) + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        if self.lm_head is None:
            return self.token_embedding.as_linear(hidden_states)
        return self.lm_head(hidden_states)


def _framework_version() -> str:
    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


def _load_portable_state(
    model: MLXDecoderOnlyTransformer, state: dict[str, np.ndarray]
) -> None:
    weights = [(name, mx.array(value)) for name, value in sorted(state.items())]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())


def _memory_value(name: str) -> int:
    function = getattr(mx, name, None)
    if not callable(function):
        return 0
    return int(function())


def _reset_peak_memory() -> None:
    function = getattr(mx, "reset_peak_memory", None)
    if callable(function):
        function()


def _export_state(
    model: MLXDecoderOnlyTransformer,
) -> dict[str, np.ndarray]:
    mx.eval(model.parameters())
    flattened = tree_flatten(model.parameters())
    if not isinstance(flattened, list):
        raise TypeError("MLX tree_flatten returned a mapping instead of a list")
    return {
        name: np.ascontiguousarray(np.array(value, copy=True))
        for name, value in flattened
    }


def run_mlx_benchmark(
    config: ExperimentConfig,
    settings: BenchmarkSettings,
    state: dict[str, np.ndarray],
    batches: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> BackendMeasurements:
    """Run an uncompiled MLX baseline on the default Apple GPU stream."""

    if config.model.dropout != 0.0:
        raise ValueError(
            "the initial cross-framework benchmark requires dropout=0"
        )
    if config.training.device not in {"auto", "mps"}:
        raise ValueError(
            "the MLX benchmark requires training.device to be auto or mps"
        )

    mx.set_default_device(mx.gpu)
    mx.random.seed(config.training.seed)
    model = MLXDecoderOnlyTransformer(config.model)
    _load_portable_state(model, state)
    state.clear()
    gc.collect()
    model.train()
    optimizer = optim.AdamW(
        learning_rate=config.training.learning_rate,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=config.training.weight_decay,
        bias_correction=True,
    )

    def loss_function(
        current_model: MLXDecoderOnlyTransformer,
        input_ids: Any,
        targets: Any,
    ) -> Any:
        logits = current_model(input_ids)
        flat_logits = mx.reshape(logits, (-1, config.model.vocab_size))
        flat_targets = mx.reshape(targets, (-1,))
        return mx.mean(nn.losses.cross_entropy(flat_logits, flat_targets))

    loss_and_grad = nn.value_and_grad(model, loss_function)
    records: list[BenchmarkStep] = []
    for index, (input_array, target_array) in enumerate(batches):
        input_ids = mx.array(input_array, dtype=mx.int32)
        targets = mx.array(target_array, dtype=mx.int32)
        mx.synchronize(mx.gpu)
        _reset_peak_memory()
        started = time.perf_counter()
        loss, gradients = loss_and_grad(model, input_ids, targets)
        if config.training.gradient_clip_norm is not None:
            gradients, _gradient_norm = optim.clip_grad_norm(
                gradients, config.training.gradient_clip_norm
            )
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state)
        mx.synchronize(mx.gpu)
        duration = time.perf_counter() - started
        available, memory_percent, swap_used = system_memory_sample()
        phase = "warmup" if index < settings.warmup_steps else "measured"
        phase_index = index if phase == "warmup" else index - settings.warmup_steps
        records.append(
            BenchmarkStep(
                phase=phase,
                index=phase_index,
                loss=float(loss.item()),
                duration_seconds=duration,
                process_rss_bytes=process_rss_bytes(),
                memory_measurement_kind="mlx-peak-memory",
                framework_memory_bytes=_memory_value("get_peak_memory"),
                driver_memory_bytes=0,
                cache_memory_bytes=_memory_value("get_cache_memory"),
                system_available_memory_bytes=available,
                system_memory_percent=memory_percent,
                system_swap_used_bytes=swap_used,
            )
        )

    return BackendMeasurements(
        framework="MLX",
        framework_version=_framework_version(),
        device=str(mx.default_device()),
        steps=tuple(records),
        memory_semantics=(
            "MLX peak and cache allocator counters. They can undercount the "
            "operating-system footprint and require external cross-checks."
        ),
        warnings=(
            "Input array creation is performed before the timed region.",
            "MLX lazy graphs are evaluated and synchronized once per step.",
            "The portable NumPy initialization arrays are released before warm-up.",
            "Final parameters are exported after the timed region.",
            "This backend is uncompiled; compiled MLX is a separate variant.",
        ),
        final_state=_export_state(model),
    )
