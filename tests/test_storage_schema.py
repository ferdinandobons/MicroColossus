from __future__ import annotations

import numpy as np
import pytest

from microcolossus.storage.codec import (
    TensorPayload,
    payload_from_numpy,
    payload_to_numpy,
)
from microcolossus.storage.schema import (
    MANIFEST_SCHEMA_VERSION,
    ChunkRecord,
    Manifest,
    TensorKind,
    TensorRecord,
)


def test_numpy_payload_normalizes_noncontiguous_big_endian_data() -> None:
    source = np.arange(24, dtype=">f4").reshape(4, 6)[:, ::2]
    assert not source.flags.c_contiguous

    payload = payload_from_numpy(source, logical_name="weights")
    restored = payload_to_numpy(payload)

    assert payload.dtype == "float32"
    assert payload.byte_order == "little"
    assert payload.shape == (4, 3)
    np.testing.assert_array_equal(restored, source.astype("<f4"))


def test_payload_validates_byte_length() -> None:
    with pytest.raises(ValueError, match="byte length mismatch"):
        TensorPayload(
            logical_name="bad",
            kind=TensorKind.PARAMETER,
            shape=(2,),
            dtype="float32",
            byte_order="little",
            data=b"1234",
        )


def test_scalar_and_zero_length_payloads_are_supported() -> None:
    scalar = payload_from_numpy(np.array(7, dtype=np.int64), logical_name="scalar")
    empty = payload_from_numpy(np.empty((0, 3), dtype=np.float32), logical_name="empty")

    assert scalar.shape == ()
    assert scalar.byte_length == 8
    assert empty.shape == (0, 3)
    assert empty.byte_length == 0
    np.testing.assert_array_equal(payload_to_numpy(scalar), np.array(7, dtype=np.int64))
    np.testing.assert_array_equal(
        payload_to_numpy(empty), np.empty((0, 3), dtype=np.float32)
    )


def test_manifest_checksum_is_deterministic_and_detects_changes() -> None:
    chunk = ChunkRecord(
        chunk_id="a" * 64,
        storage_path="chunks/aa/" + "a" * 64 + ".chunk",
        byte_offset=0,
        byte_length=4,
        checksum="a" * 64,
        compression="none",
        creating_transaction="tx-1",
    )
    tensor = TensorRecord(
        tensor_id="parameter:w",
        logical_name="w",
        kind=TensorKind.PARAMETER,
        shape=(1,),
        dtype="float32",
        byte_order="little",
        version=0,
        chunk_ids=(chunk.chunk_id,),
        byte_length=4,
        checksum="b" * 64,
        committed_step=0,
    )
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        manifest_id="manifest-1",
        parent_manifest_id=None,
        committed_step=0,
        created_at_utc="2026-07-27T00:00:00Z",
        tensors=(tensor,),
        chunks=(chunk,),
        aggregate_logical_bytes=4,
        aggregate_physical_bytes=4,
    ).with_computed_checksum()

    assert manifest.compute_checksum() == manifest.manifest_checksum
    assert Manifest.from_dict(manifest.to_dict()) == manifest
    modified = Manifest.from_dict({**manifest.to_dict(), "committed_step": 1})
    with pytest.raises(ValueError, match="checksum mismatch"):
        modified.validate_checksum()
