from __future__ import annotations

from pathlib import Path

import pytest
import torch

from microcolossus.storage import (
    StoreLimits,
    VersionedTensorStore,
    export_pytorch_state,
    restore_pytorch_state,
)


def _trained_model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.AdamW]:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_pytorch_model_and_adamw_round_trip_through_store(tmp_path: Path) -> None:
    model, optimizer = _trained_model_and_optimizer()
    payloads = export_pytorch_state(model, optimizer)
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(max_storage_bytes=10_000_000),
    )
    transaction = store.begin_transaction(committed_step=0)
    tensor_ids = transaction.put_many(payloads)
    transaction.commit()

    torch.manual_seed(99)
    restored_model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_payloads = [store.read_tensor(tensor_id) for tensor_id in tensor_ids]
    restore_pytorch_state(restored_model, restored_payloads, restored_optimizer)

    for original, restored in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
    original_names = dict(model.named_parameters(remove_duplicate=True))
    restored_names = dict(restored_model.named_parameters(remove_duplicate=True))
    for name in original_names:
        original_state = optimizer.state[original_names[name]]
        restored_state = restored_optimizer.state[restored_names[name]]
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            original_value = original_state[key]
            restored_value = restored_state[key]
            if isinstance(original_value, torch.Tensor):
                torch.testing.assert_close(original_value, restored_value, rtol=0, atol=0)
            else:
                assert original_value == restored_value


def test_mlx_adapter_is_optional() -> None:
    pytest.importorskip("mlx")
