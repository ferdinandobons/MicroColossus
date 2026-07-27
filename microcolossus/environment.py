"""Environment inspection for reproducible CPU, CUDA, and MPS diagnostics."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch


@dataclass(frozen=True)
class EnvironmentReport:
    platform: str
    machine: str
    python_version: str
    torch_version: str
    system_memory_bytes: int
    cuda_available: bool
    cuda_version: str | None
    mps_built: bool
    mps_available: bool
    mps_device_name: str | None
    mps_core_count: int | None
    mps_recommended_max_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_backend_value(name: str) -> Any:
    backend = getattr(torch.backends, "mps", None)
    function = getattr(backend, name, None) if backend is not None else None
    if not callable(function):
        return None
    try:
        return function()
    except (AttributeError, NotImplementedError, RuntimeError):
        return None


def _optional_mps_value(name: str) -> int | None:
    namespace = getattr(torch, "mps", None)
    function = getattr(namespace, name, None) if namespace is not None else None
    if not callable(function):
        return None
    try:
        return int(function())
    except (AttributeError, NotImplementedError, RuntimeError):
        return None


def collect_environment() -> EnvironmentReport:
    """Collect a JSON-serializable accelerator environment report."""

    backend = getattr(torch.backends, "mps", None)
    mps_built = bool(backend is not None and backend.is_built())
    mps_available = bool(backend is not None and backend.is_available())
    device_name = _optional_backend_value("get_name") if mps_available else None
    core_count = _optional_backend_value("get_core_count") if mps_available else None
    return EnvironmentReport(
        platform=platform.platform(),
        machine=platform.machine(),
        python_version=sys.version.split()[0],
        torch_version=str(torch.__version__),
        system_memory_bytes=int(psutil.virtual_memory().total),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_version=torch.version.cuda,
        mps_built=mps_built,
        mps_available=mps_available,
        mps_device_name=str(device_name) if device_name is not None else None,
        mps_core_count=int(core_count) if core_count is not None else None,
        mps_recommended_max_memory_bytes=(
            _optional_mps_value("recommended_max_memory") if mps_available else None
        ),
    )
