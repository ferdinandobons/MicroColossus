"""Canonical tensor byte encoding independent of PyTorch and MLX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schema import TensorKind, sha256_hex

_DTYPE_SIZES: dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def dtype_size(dtype: str) -> int:
    try:
        return _DTYPE_SIZES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported canonical dtype: {dtype}") from exc


def element_count(shape: tuple[int, ...]) -> int:
    if any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions cannot be negative")
    return math.prod(shape, start=1)


def expected_byte_length(shape: tuple[int, ...], dtype: str) -> int:
    return element_count(shape) * dtype_size(dtype)


@dataclass(frozen=True)
class TensorPayload:
    """Canonical logical tensor supplied to a store transaction."""

    logical_name: str
    kind: TensorKind
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    data: bytes
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("logical_name cannot be empty")
        if self.dtype not in _DTYPE_SIZES:
            raise ValueError(f"unsupported canonical dtype: {self.dtype}")
        expected = expected_byte_length(self.shape, self.dtype)
        if len(self.data) != expected:
            raise ValueError(
                f"tensor byte length mismatch for {self.logical_name}: "
                f"{len(self.data)} != {expected}"
            )
        expected_order = "not_applicable" if dtype_size(self.dtype) == 1 else "little"
        if self.byte_order != expected_order:
            raise ValueError(
                f"canonical byte order for {self.dtype} must be {expected_order}"
            )

    @property
    def checksum(self) -> str:
        return sha256_hex(self.data)

    @property
    def byte_length(self) -> int:
        return len(self.data)

    def stable_tensor_id(self) -> str:
        return f"{self.kind.value}:{self.logical_name}"


def payload_from_numpy(
    value: Any,
    *,
    logical_name: str,
    kind: TensorKind = TensorKind.PARAMETER,
    metadata: tuple[tuple[str, str], ...] = (),
) -> TensorPayload:
    """Normalize a NumPy-compatible array to canonical little-endian bytes."""

    import numpy as np

    array = np.asarray(value)
    canonical_dtype = array.dtype.name
    if canonical_dtype not in _DTYPE_SIZES:
        raise ValueError(f"unsupported NumPy dtype: {array.dtype}")
    target_dtype = array.dtype
    if array.dtype.itemsize > 1:
        target_dtype = array.dtype.newbyteorder("<")
    normalized = np.array(array, dtype=target_dtype, order="C", copy=True)
    byte_order = "not_applicable" if normalized.dtype.itemsize == 1 else "little"
    return TensorPayload(
        logical_name=logical_name,
        kind=kind,
        shape=tuple(int(item) for item in normalized.shape),
        dtype=canonical_dtype,
        byte_order=byte_order,
        data=normalized.tobytes(order="C"),
        metadata=metadata,
    )


def payload_to_numpy(payload: TensorPayload, *, copy: bool = True) -> Any:
    """Decode canonical bytes as a NumPy array."""

    import numpy as np

    if payload.dtype == "bfloat16":
        raise ValueError("NumPy does not provide a portable built-in bfloat16 dtype")
    dtype = np.dtype(payload.dtype)
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    array = np.frombuffer(payload.data, dtype=dtype).reshape(payload.shape)
    return array.copy() if copy else array
