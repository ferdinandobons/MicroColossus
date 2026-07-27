from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcolossus.step_bundle import (
    BundleFailurePoint,
    BundleSimulatedCrash,
    StepBundleStore,
)
from microcolossus.storage import StoreLimits, TensorKind, TensorPayload, VersionedTensorStore


def _payload(name: str, value: bytes) -> TensorPayload:
    return TensorPayload(
        logical_name=name,
        kind=TensorKind.PARAMETER,
        shape=(len(value),),
        dtype="uint8",
        byte_order="not_applicable",
        data=value,
    )


def _child_store(path: Path, name: str, value: bytes) -> VersionedTensorStore:
    store = VersionedTensorStore.create(
        path,
        limits=StoreLimits(
            chunk_size_bytes=1024,
            max_storage_bytes=1024**2,
            max_staging_bytes=1024,
        ),
    )
    transaction = store.begin_transaction(committed_step=0)
    transaction.put_tensor(_payload(name, value))
    transaction.commit()
    return store


def test_step_bundle_publish_verify_and_recover(tmp_path: Path) -> None:
    bundle = StepBundleStore.create(tmp_path / "bundle")
    parameters = _child_store(bundle.root / "work" / "parameters", "model.weight", b"abc")
    optimizer = _child_store(bundle.root / "work" / "optimizer", "optimizer.state", b"def")

    initial, _ = bundle.publish(
        committed_step=0,
        parameter_store_path=parameters.root,
        optimizer_store_path=optimizer.root,
        gradient_store_path=None,
        batch_checksum="batch-0",
    )

    verification = bundle.verify()
    recovery = bundle.recover()
    assert verification.bundle_id == initial.bundle_id
    assert verification.committed_step == 0
    assert verification.parameter_tensor_count == 1
    assert verification.optimizer_tensor_count == 1
    assert recovery.current_bundle_id == initial.bundle_id
    assert not recovery.unpublished_bundle_ids


def test_step_bundle_failure_before_current_keeps_previous_bundle(tmp_path: Path) -> None:
    bundle = StepBundleStore.create(tmp_path / "bundle")
    parameters = _child_store(bundle.root / "work" / "parameters", "model.weight", b"abc")
    optimizer = _child_store(bundle.root / "work" / "optimizer", "optimizer.state", b"def")
    initial, _ = bundle.publish(
        committed_step=0,
        parameter_store_path=parameters.root,
        optimizer_store_path=optimizer.root,
        gradient_store_path=None,
        batch_checksum="batch-0",
    )
    candidate_parameters = _child_store(
        bundle.root / "candidates" / "parameters", "model.weight", b"xyz"
    )
    candidate_optimizer = _child_store(
        bundle.root / "candidates" / "optimizer", "optimizer.state", b"uvw"
    )

    def fail(point: BundleFailurePoint, context: dict[str, object]) -> None:
        del context
        if point is BundleFailurePoint.BEFORE_CURRENT_RENAME:
            raise BundleSimulatedCrash("stop before CURRENT replacement")

    with pytest.raises(BundleSimulatedCrash):
        bundle.publish(
            committed_step=1,
            parameter_store_path=candidate_parameters.root,
            optimizer_store_path=candidate_optimizer.root,
            gradient_store_path=None,
            batch_checksum="batch-1",
            failure_injector=fail,
        )

    assert bundle.current_manifest().bundle_id == initial.bundle_id
    recovery = bundle.recover()
    assert recovery.current_bundle_id == initial.bundle_id
    assert recovery.unpublished_bundle_ids


def test_step_bundle_checksum_detects_manifest_corruption(tmp_path: Path) -> None:
    bundle = StepBundleStore.create(tmp_path / "bundle")
    parameters = _child_store(bundle.root / "work" / "parameters", "model.weight", b"abc")
    optimizer = _child_store(bundle.root / "work" / "optimizer", "optimizer.state", b"def")
    initial, _ = bundle.publish(
        committed_step=0,
        parameter_store_path=parameters.root,
        optimizer_store_path=optimizer.root,
        gradient_store_path=None,
        batch_checksum="batch-0",
    )
    manifest_path = bundle.root / "manifests" / f"{initial.bundle_id}.json"
    value = json.loads(manifest_path.read_text())
    value["batch_checksum"] = "corrupt"
    manifest_path.write_text(json.dumps(value))

    with pytest.raises(RuntimeError, match="checksum"):
        bundle.current_manifest()
