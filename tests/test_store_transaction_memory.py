from __future__ import annotations

import gc
import weakref
from pathlib import Path

from microcolossus.storage import TensorKind, TensorPayload, VersionedTensorStore


def _payload() -> TensorPayload:
    return TensorPayload(
        logical_name="model.weight",
        kind=TensorKind.PARAMETER,
        shape=(1024,),
        dtype="uint8",
        byte_order="not_applicable",
        data=b"x" * 1024,
    )


def test_successful_transaction_releases_staged_payloads(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(tmp_path / "store")
    transaction = store.begin_transaction(committed_step=0)
    payload = _payload()
    reference = weakref.ref(payload)
    transaction.put_tensor(payload, version=0)
    del payload

    assert reference() is not None
    transaction.commit()
    gc.collect()

    assert reference() is None
    assert transaction._payloads == {}
    assert transaction._explicit_versions == {}
    assert store.verify().tensor_count == 1


def test_aborted_transaction_releases_staged_payloads(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(tmp_path / "store")
    transaction = store.begin_transaction(committed_step=0)
    payload = _payload()
    reference = weakref.ref(payload)
    transaction.put_tensor(payload, version=0)
    del payload

    assert reference() is not None
    transaction.abort("memory-release regression test")
    gc.collect()

    assert reference() is None
    assert transaction._payloads == {}
    assert transaction._explicit_versions == {}
    assert store.current_manifest().committed_step == -1
