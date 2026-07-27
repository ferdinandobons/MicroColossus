"""Controlled decoder-only Transformer used by the numerical baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


@dataclass
class CausalLMOutput:
    """Output produced by :class:`DecoderOnlyTransformer`."""

    logits: Tensor
    loss: Tensor | None


class CausalSelfAttention(nn.Module):
    """Reference causal self-attention without custom kernels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_size = config.hidden_size // config.heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, sequence_length, self.heads, self.head_size).transpose(
                1, 2
            )

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden_states.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)

        context = probabilities @ value
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, hidden_size
        )
        return self.residual_dropout(self.output(context))


class FeedForward(nn.Module):
    """Reference Transformer MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        intermediate_size = config.hidden_size * config.mlp_ratio
        self.input = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.output = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.output(F.gelu(self.input(hidden_states), approximate="tanh")))


class TransformerBlock(nn.Module):
    """Pre-normalization Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class DecoderOnlyTransformer(nn.Module):
    """Small, explicit causal language model used as the resident oracle."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds model.max_sequence_length")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return CausalLMOutput(logits=logits, loss=loss)

    @property
    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
