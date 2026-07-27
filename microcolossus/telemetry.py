"""Telemetry primitives shared by training, planning, and diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import psutil
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AcceleratorMemoryMetrics:
    """One synchronized accelerator-memory sample."""

    measurement_kind: str
    allocated_bytes: int
    driver_allocated_bytes: int
    recommended_max_bytes: int


def process_rss_bytes() -> int:
    """Return current process resident memory."""

    return int(psutil.Process(os.getpid()).memory_info().rss)


def _call_int(namespace: object, name: str) -> int:
    function = getattr(namespace, name, None)
    if not callable(function):
        return 0
    return int(function())


def accelerator_memory_metrics(device: torch.device) -> AcceleratorMemoryMetrics:
    """Measure accelerator memory using backend-specific public APIs.

    CUDA reports the peak tensor allocation since the last reset. MPS exposes a
    synchronized current allocation rather than a resettable peak, plus the
    total allocation attributed to the Metal driver and its recommended working
    set. These measurements are not directly interchangeable.
    """

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        return AcceleratorMemoryMetrics(
            measurement_kind="cuda-peak-allocated",
            allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            driver_allocated_bytes=0,
            recommended_max_bytes=int(properties.total_memory),
        )
    if device.type == "mps":
        return AcceleratorMemoryMetrics(
            measurement_kind="mps-current-allocated",
            allocated_bytes=_call_int(torch.mps, "current_allocated_memory"),
            driver_allocated_bytes=_call_int(torch.mps, "driver_allocated_memory"),
            recommended_max_bytes=_call_int(torch.mps, "recommended_max_memory"),
        )
    return AcceleratorMemoryMetrics(
        measurement_kind="none",
        allocated_bytes=0,
        driver_allocated_bytes=0,
        recommended_max_bytes=0,
    )


def reset_accelerator_memory(device: torch.device) -> None:
    """Reset a backend peak counter when the backend exposes one."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_accelerator(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before timing or sampling."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _tensor_bytes(tensor: Tensor) -> bytes:
    """Return exact logical tensor bytes without requiring NumPy."""

    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch.uint8)
    storage = bytes(cast(Iterable[int], byte_view.untyped_storage()))
    start = byte_view.storage_offset()
    return storage[start : start + byte_view.numel()]


def model_checksum(model: nn.Module) -> str:
    """Create a deterministic SHA-256 checksum of the model state."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _json_serializable(value: Any) -> Any:
    if not isinstance(value, type) and is_dataclass(value):
        return asdict(cast(Any, value))
    return value


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Write JSON through a temporary file and atomic rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = _json_serializable(value)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


class JsonlWriter:
    """Append-only JSON Lines writer for step telemetry."""

    def __init__(self, path: str | Path, *, truncate: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            self.path.unlink(missing_ok=True)

    def append(self, value: Any) -> None:
        serializable = _json_serializable(value)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(serializable, handle, sort_keys=True)
            handle.write("\n")
