import torch

import microcolossus.environment as environment_module
from microcolossus.environment import collect_environment


def test_optional_mps_metadata_remains_none_when_api_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(environment_module, "_optional_backend_value", lambda _name: None)
    monkeypatch.setattr(environment_module, "_optional_mps_value", lambda _name: None)

    report = collect_environment()

    assert report.mps_built
    assert report.mps_available
    assert report.mps_device_name is None
    assert report.mps_core_count is None
    assert report.mps_recommended_max_memory_bytes is None
