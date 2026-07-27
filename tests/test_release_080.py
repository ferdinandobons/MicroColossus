from __future__ import annotations

import microcolossus


def test_bounded_step_release_version() -> None:
    assert microcolossus.__version__ == "0.8.0"
