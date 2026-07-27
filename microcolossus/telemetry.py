"""Small telemetry primitives shared by training and planning."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import Tensor, nn


def process_rss_bytes() -> int:
    """Return current process resident memory."""

    return int(psutil.Process(os.getpid()).memory_info().rss)


def peak_vram_bytes(device: torch.device) -> int:
    """Return CUDA peak allocation for the current measurement window."""

    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def reset_peak_vram(device: torch.device) -> None:
    """Reset CUDA peak allocation when CUDA is active."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.int16).numpy().tobytes()
    return value.numpy().tobytes()


def model_checksum(model: nn.Module) -> str:
    """Create a deterministic SHA-256 checksum of the model state."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Write JSON through a temporary file and atomic rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = asdict(value) if is_dataclass(value) else value
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
        serializable = asdict(value) if is_dataclass(value) else value
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(serializable, handle, sort_keys=True)
            handle.write("\n")
