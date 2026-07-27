import tomllib
from pathlib import Path

import microcolossus


def _project_metadata() -> dict[str, object]:
    with Path("pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_package_version_matches_project_metadata() -> None:
    project = _project_metadata()
    assert microcolossus.__version__ == project["version"]


def test_numpy_extras_preserve_python_311_typing_compatibility() -> None:
    project = _project_metadata()
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    for extra_name in ("dev", "benchmark"):
        requirements = extras[extra_name]
        assert isinstance(requirements, list)
        assert "numpy>=2.4,<2.5" in requirements
