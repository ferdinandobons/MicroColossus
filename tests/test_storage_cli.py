from __future__ import annotations

import json
from pathlib import Path

from microcolossus.cli import main


def test_storage_cli_lifecycle(tmp_path: Path, capsys) -> None:
    path = tmp_path / "store"
    result = main(
        [
            "store-init",
            "--path",
            str(path),
            "--chunk-size-mib",
            "1",
            "--max-staging-mib",
            "1",
            "--max-storage-gib",
            "0.01",
        ]
    )
    created = json.loads(capsys.readouterr().out)
    assert result == 0
    assert created["committed_step"] == -1

    result = main(["store-verify", "--path", str(path)])
    verified = json.loads(capsys.readouterr().out)
    assert result == 0
    assert verified["tensor_count"] == 0

    result = main(["store-recover", "--path", str(path)])
    recovered = json.loads(capsys.readouterr().out)
    assert result == 0
    assert recovered["incomplete_transactions"] == []
