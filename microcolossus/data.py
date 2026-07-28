"""Deterministic synthetic and local UTF-8 text data frontends."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .config import ExperimentConfig
from .storage.schema import canonical_json_bytes, sha256_hex

DATA_IDENTITY_SCHEMA_VERSION = "microcolossus.data-identity.v1"
UTF8_BYTE_TOKENIZER_VERSION = "utf8-bytes-v1"
SYNTHETIC_TOKENIZER_VERSION = "synthetic-token-ids-v1"
SYNTHETIC_BATCH_STREAM_VERSION = "synthetic-seed-per-cursor-v1"
TEXT_BATCH_STREAM_VERSION = "utf8-byte-random-window-v1"
VALIDATION_SEED_OFFSET = 10_000_000


@dataclass(frozen=True)
class DataIdentity:
    """Checksummed identity of the source, tokenizer, split, and sampler."""

    schema_version: str
    source_kind: str
    tokenizer_version: str
    batch_stream_version: str
    split_kind: str
    train_checksum: str
    validation_checksum: str
    train_byte_count: int
    validation_byte_count: int
    identity_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "tokenizer_version": self.tokenizer_version,
            "batch_stream_version": self.batch_stream_version,
            "split_kind": self.split_kind,
            "train_checksum": self.train_checksum,
            "validation_checksum": self.validation_checksum,
            "train_byte_count": self.train_byte_count,
            "validation_byte_count": self.validation_byte_count,
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_checksum(self) -> DataIdentity:
        return DataIdentity(
            schema_version=self.schema_version,
            source_kind=self.source_kind,
            tokenizer_version=self.tokenizer_version,
            batch_stream_version=self.batch_stream_version,
            split_kind=self.split_kind,
            train_checksum=self.train_checksum,
            validation_checksum=self.validation_checksum,
            train_byte_count=self.train_byte_count,
            validation_byte_count=self.validation_byte_count,
            identity_checksum=self.compute_checksum(),
        )

    def validate(self) -> None:
        if self.schema_version != DATA_IDENTITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported data identity schema: {self.schema_version}")
        if self.identity_checksum != self.compute_checksum():
            raise ValueError("data identity checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["identity_checksum"] = self.identity_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DataIdentity:
        result = cls(
            schema_version=str(value["schema_version"]),
            source_kind=str(value["source_kind"]),
            tokenizer_version=str(value["tokenizer_version"]),
            batch_stream_version=str(value["batch_stream_version"]),
            split_kind=str(value["split_kind"]),
            train_checksum=str(value["train_checksum"]),
            validation_checksum=str(value["validation_checksum"]),
            train_byte_count=int(value["train_byte_count"]),
            validation_byte_count=int(value["validation_byte_count"]),
            identity_checksum=str(value["identity_checksum"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class LanguageModelBatch:
    """One deterministic next-token batch plus cursor provenance."""

    input_ids: Tensor
    targets: Tensor
    cursor: int
    seed: int
    split: str
    offsets: tuple[int, ...]
    source_kind: str


class PreparedDataSource(Protocol):
    """Runtime interface consumed by resident and bounded training."""

    identity: DataIdentity

    def training_batch(self, cursor: int) -> LanguageModelBatch: ...

    def validation_batch(self, cursor: int) -> LanguageModelBatch: ...

    def encode_text(self, text: str) -> tuple[int, ...]: ...

    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str: ...

    def default_prompt(self, max_tokens: int) -> str: ...


class SyntheticDataSource:
    """Existing deterministic random-token source."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.identity = DataIdentity(
            schema_version=DATA_IDENTITY_SCHEMA_VERSION,
            source_kind="synthetic",
            tokenizer_version=SYNTHETIC_TOKENIZER_VERSION,
            batch_stream_version=SYNTHETIC_BATCH_STREAM_VERSION,
            split_kind="synthetic-independent-streams-v1",
            train_checksum="",
            validation_checksum="",
            train_byte_count=0,
            validation_byte_count=0,
        ).with_checksum()

    def _batch(self, cursor: int, *, split: str) -> LanguageModelBatch:
        if cursor < 0:
            raise ValueError("batch cursor cannot be negative")
        offset = 1 if split == "train" else VALIDATION_SEED_OFFSET
        seed = self.config.training.seed + offset + cursor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        tokens = torch.randint(
            low=0,
            high=self.config.model.vocab_size,
            size=(
                self.config.training.micro_batch_size,
                self.config.training.sequence_length + 1,
            ),
            generator=generator,
            dtype=torch.long,
        )
        input_ids = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        return LanguageModelBatch(
            input_ids=input_ids,
            targets=targets,
            cursor=cursor,
            seed=seed,
            split=split,
            offsets=(),
            source_kind="synthetic",
        )

    def training_batch(self, cursor: int) -> LanguageModelBatch:
        return self._batch(cursor, split="train")

    def validation_batch(self, cursor: int) -> LanguageModelBatch:
        return self._batch(cursor, split="validation")

    def encode_text(self, text: str) -> tuple[int, ...]:
        raise RuntimeError("synthetic data does not expose text tokenization")

    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        raise RuntimeError("synthetic data does not expose text decoding")

    def default_prompt(self, max_tokens: int) -> str:
        del max_tokens
        return ""


class Utf8ByteTextDataSource:
    """Local text source tokenized as exact UTF-8 bytes."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        train_bytes: bytes,
        validation_bytes: bytes,
        split_kind: str,
    ) -> None:
        self.config = config
        self._train_bytes = train_bytes
        self._validation_bytes = validation_bytes
        self.identity = DataIdentity(
            schema_version=DATA_IDENTITY_SCHEMA_VERSION,
            source_kind="utf8_text",
            tokenizer_version=UTF8_BYTE_TOKENIZER_VERSION,
            batch_stream_version=TEXT_BATCH_STREAM_VERSION,
            split_kind=split_kind,
            train_checksum=hashlib.sha256(train_bytes).hexdigest(),
            validation_checksum=hashlib.sha256(validation_bytes).hexdigest(),
            train_byte_count=len(train_bytes),
            validation_byte_count=len(validation_bytes),
        ).with_checksum()

    @staticmethod
    def encode_text(text: str) -> tuple[int, ...]:
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode_tokens(token_ids: tuple[int, ...] | list[int]) -> str:
        return bytes(int(item) & 0xFF for item in token_ids).decode(
            "utf-8", errors="replace"
        )

    def default_prompt(self, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        prefix = self._validation_bytes[:max_tokens]
        return prefix.decode("utf-8", errors="replace")

    def _batch(self, data: bytes, cursor: int, *, split: str) -> LanguageModelBatch:
        if cursor < 0:
            raise ValueError("batch cursor cannot be negative")
        sequence_length = self.config.training.sequence_length
        window_length = sequence_length + 1
        max_offset = len(data) - window_length
        if max_offset < 0:
            raise ValueError(
                f"{split} corpus must contain at least {window_length} UTF-8 bytes"
            )
        seed_offset = 1 if split == "train" else VALIDATION_SEED_OFFSET
        seed = self.config.training.seed + seed_offset + cursor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        offsets_tensor = torch.randint(
            low=0,
            high=max_offset + 1,
            size=(self.config.training.micro_batch_size,),
            generator=generator,
            dtype=torch.int64,
        )
        offsets = tuple(int(item) for item in offsets_tensor.tolist())
        rows = [list(data[offset : offset + window_length]) for offset in offsets]
        tokens = torch.tensor(rows, dtype=torch.long)
        return LanguageModelBatch(
            input_ids=tokens[:, :-1].contiguous(),
            targets=tokens[:, 1:].contiguous(),
            cursor=cursor,
            seed=seed,
            split=split,
            offsets=offsets,
            source_kind="utf8_text",
        )

    def training_batch(self, cursor: int) -> LanguageModelBatch:
        return self._batch(self._train_bytes, cursor, split="train")

    def validation_batch(self, cursor: int) -> LanguageModelBatch:
        return self._batch(self._validation_bytes, cursor, split="validation")


def _read_bytes(path_value: str | None, name: str) -> bytes:
    if path_value is None:
        raise ValueError(f"{name} path is required")
    path = Path(path_value)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {name} corpus: {path}") from exc
    if not data:
        raise ValueError(f"{name} corpus is empty: {path}")
    return data


def _split_single_corpus(config: ExperimentConfig, data: bytes) -> tuple[bytes, bytes]:
    minimum = config.training.sequence_length + 1
    if len(data) < 2 * minimum:
        raise ValueError(
            "single-file utf8_text corpus is too small for deterministic train/validation split"
        )
    validation_size = max(minimum, int(round(len(data) * config.data.validation_fraction)))
    validation_size = min(validation_size, len(data) - minimum)
    split_at = len(data) - validation_size
    return data[:split_at], data[split_at:]


def prepare_data_source(config: ExperimentConfig) -> PreparedDataSource:
    """Prepare one immutable data source for a training invocation."""

    if config.data.kind == "synthetic":
        return SyntheticDataSource(config)
    train_file = _read_bytes(config.data.train_path, "train")
    if config.data.validation_path is None:
        train_bytes, validation_bytes = _split_single_corpus(config, train_file)
        split_kind = "single-file-tail-fraction-v1"
    else:
        train_bytes = train_file
        validation_bytes = _read_bytes(config.data.validation_path, "validation")
        split_kind = "separate-files-v1"
    minimum = config.training.sequence_length + 1
    if len(train_bytes) < minimum or len(validation_bytes) < minimum:
        raise ValueError(f"each text split must contain at least {minimum} UTF-8 bytes")
    return Utf8ByteTextDataSource(
        config,
        train_bytes=train_bytes,
        validation_bytes=validation_bytes,
        split_kind=split_kind,
    )
