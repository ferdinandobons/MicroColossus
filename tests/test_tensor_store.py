from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microcolossus.storage import (
    BudgetExceededError,
    FailurePoint,
    IntegrityError,
    SimulatedCrash,
    StoreLimits,
    TensorKind,
    VersionedTensorStore,
    payload_from_numpy,
    payload_to_numpy,
)


def _payload(name: str, values: np.ndarray, kind: TensorKind = TensorKind.PARAMETER):
    return payload_from_numpy(values, logical_name=name, kind=kind)


def test_store_round_trip_and_chunk_range(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(
            chunk_size_bytes=16,
            max_staging_bytes=16,
            max_storage_bytes=2_000_000,
        ),
    )
    values = np.arange(20, dtype=np.float32)
    transaction = store.begin_transaction(committed_step=0)
    tensor_id = transaction.put_tensor(_payload("model.weight", values))
    result = transaction.commit()

    assert result.manifest.committed_step == 0
    assert len(result.manifest.chunks) == 5
    assert result.telemetry.chunk_writes == 5
    restored = store.read_tensor(tensor_id)
    np.testing.assert_array_equal(payload_to_numpy(restored), values)
    assert store.read_tensor_range(tensor_id, start=4, length=8) == values.tobytes()[4:12]
    report = store.verify()
    assert report.tensor_count == 1
    assert report.chunk_count == 5


def test_copy_on_write_reuses_unchanged_chunks_and_versions(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(
            chunk_size_bytes=8,
            max_staging_bytes=8,
            max_storage_bytes=2_000_000,
        ),
    )
    first = store.begin_transaction(committed_step=0)
    first.put_tensor(_payload("a", np.arange(4, dtype=np.float32)))
    first.put_tensor(_payload("b", np.arange(4, dtype=np.float32) + 10))
    first_result = first.commit()

    second = store.begin_transaction(committed_step=1)
    second.put_tensor(_payload("a", np.arange(4, dtype=np.float32) + 1))
    second_result = second.commit()

    first_records = {item.tensor_id: item for item in first_result.manifest.tensors}
    second_records = {item.tensor_id: item for item in second_result.manifest.tensors}
    assert second_records["parameter:a"].version == 1
    assert second_records["parameter:b"].version == 0
    assert second_records["parameter:b"].chunk_ids == first_records["parameter:b"].chunk_ids
    np.testing.assert_array_equal(
        payload_to_numpy(store.read_tensor("parameter:b")),
        np.arange(4, dtype=np.float32) + 10,
    )


def test_corruption_is_detected_before_returning_tensor(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(max_storage_bytes=2_000_000),
    )
    transaction = store.begin_transaction(committed_step=0)
    transaction.put_tensor(_payload("w", np.arange(8, dtype=np.float32)))
    result = transaction.commit()
    chunk = result.manifest.chunks[0]
    path = store.root / chunk.storage_path
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(IntegrityError, match="integrity failure"):
        store.read_tensor("parameter:w")
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        store.verify()


def test_storage_budget_rejects_transaction_before_publication(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(
            chunk_size_bytes=64,
            max_staging_bytes=64,
            max_storage_bytes=30_000,
        ),
    )
    current = store.current_manifest_id()
    transaction = store.begin_transaction(committed_step=0)
    transaction.put_tensor(_payload("large", np.arange(100_000, dtype=np.float32)))

    with pytest.raises(BudgetExceededError):
        transaction.commit()
    assert store.current_manifest_id() == current


def test_staging_budget_is_validated_by_limits() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        StoreLimits(chunk_size_bytes=1024, max_staging_bytes=512)


def test_recovery_after_partial_chunk_write_keeps_previous_manifest(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(
            chunk_size_bytes=64,
            max_staging_bytes=64,
            max_storage_bytes=2_000_000,
        ),
    )
    original = store.current_manifest_id()

    def crash(point: FailurePoint, _context: dict[str, object]) -> None:
        if point is FailurePoint.DURING_CHUNK_WRITE:
            raise SimulatedCrash("partial write")

    transaction = store.begin_transaction(committed_step=0, failure_injector=crash)
    transaction.put_tensor(_payload("w", np.arange(64, dtype=np.float32)))
    with pytest.raises(SimulatedCrash):
        transaction.commit()

    reopened = VersionedTensorStore.open(store.root)
    recovery = reopened.recover()
    assert reopened.current_manifest_id() == original
    assert transaction.transaction_id in recovery.incomplete_transactions
    assert transaction.transaction_id in recovery.aborted_transactions
    assert any(path.endswith(".part") for path in recovery.orphan_paths)


def test_recovery_after_unpublished_manifest_keeps_previous_manifest(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(max_storage_bytes=2_000_000),
    )
    original = store.current_manifest_id()

    def crash(point: FailurePoint, _context: dict[str, object]) -> None:
        if point is FailurePoint.BEFORE_CURRENT_RENAME:
            raise SimulatedCrash("manifest written but not published")

    transaction = store.begin_transaction(committed_step=0, failure_injector=crash)
    transaction.put_tensor(_payload("w", np.arange(8, dtype=np.float32)))
    with pytest.raises(SimulatedCrash):
        transaction.commit()

    reopened = VersionedTensorStore.open(store.root)
    recovery = reopened.recover()
    assert reopened.current_manifest_id() == original
    assert transaction.candidate_manifest_id in recovery.unpublished_manifests
    assert transaction.transaction_id in recovery.aborted_transactions


def test_zero_length_tensor_round_trip(tmp_path: Path) -> None:
    store = VersionedTensorStore.create(
        tmp_path / "store",
        limits=StoreLimits(max_storage_bytes=2_000_000),
    )
    transaction = store.begin_transaction(committed_step=0)
    transaction.put_tensor(_payload("empty", np.empty((0, 2), dtype=np.float32)))
    result = transaction.commit()

    record = next(item for item in result.manifest.tensors if item.logical_name == "empty")
    assert record.chunk_ids == ()
    restored = store.read_tensor(record.tensor_id)
    assert restored.byte_length == 0
    assert restored.shape == (0, 2)


@pytest.mark.parametrize(
    "failure_point",
    [
        FailurePoint.BEFORE_CHUNK_WRITE,
        FailurePoint.BEFORE_CHUNK_FSYNC,
        FailurePoint.BEFORE_MANIFEST_RENAME,
        FailurePoint.BEFORE_CURRENT_RENAME,
    ],
)
def test_every_injected_interruption_preserves_current_manifest(
    tmp_path: Path, failure_point: FailurePoint
) -> None:
    store = VersionedTensorStore.create(
        tmp_path / failure_point.value,
        limits=StoreLimits(
            chunk_size_bytes=64,
            max_staging_bytes=64,
            max_storage_bytes=2_000_000,
        ),
    )
    original = store.current_manifest_id()

    def crash(point: FailurePoint, _context: dict[str, object]) -> None:
        if point is failure_point:
            raise SimulatedCrash(failure_point.value)

    transaction = store.begin_transaction(committed_step=0, failure_injector=crash)
    transaction.put_tensor(_payload("w", np.arange(64, dtype=np.float32)))
    with pytest.raises(SimulatedCrash):
        transaction.commit()

    reopened = VersionedTensorStore.open(store.root)
    report = reopened.recover()
    assert reopened.current_manifest_id() == original
    assert transaction.transaction_id in report.aborted_transactions
