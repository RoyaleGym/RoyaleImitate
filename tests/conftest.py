"""What RoyaleImitate's tests share: the MockEngine environment and the update rectangle from
``royalelearn.testing``, and this package installed the way ``pip install -e`` installs it.

The package's two config sections are found by RoyaleLearn through entry points, which are
install metadata. The session writes that metadata for this checkout -- a ``.dist-info`` folder
with the entry points pyproject.toml declares and a ``direct_url.json`` naming this checkout --
onto the front of ``sys.path``, so a test runs the discovery a user's install runs, whether or
not this package is installed in the environment.
"""

from __future__ import annotations

import importlib
import json
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def declared_entry_points() -> dict[str, str]:
    """The ``royalelearn.extensions`` entry points pyproject.toml declares."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return dict(project["entry-points"]["royalelearn.extensions"])


@pytest.fixture(scope="session", autouse=True)
def installed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    site = tmp_path_factory.mktemp("site")
    info = site / "royaleimitate-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: royaleimitate\nVersion: 0.1.0\n", encoding="utf-8"
    )
    points = ["[royalelearn.extensions]"] + [
        f"{key} = {value}" for key, value in declared_entry_points().items()
    ]
    (info / "entry_points.txt").write_text("\n".join(points) + "\n", encoding="utf-8")
    direct = {"url": ROOT.as_uri(), "dir_info": {"editable": True}}
    (info / "direct_url.json").write_text(json.dumps(direct), encoding="utf-8")
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    try:
        yield site
    finally:
        sys.path.remove(str(site))
        importlib.invalidate_caches()


@pytest.fixture(scope="session")
def mock_env_spec() -> Any:
    from royalelearn.testing import mock_env_spec

    return mock_env_spec()


@pytest.fixture(scope="session")
def env_spec(mock_env_spec: Any) -> Any:
    from royalelearn.testing import read_env_spec

    return read_env_spec(mock_env_spec)


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, Any]]:
    from royalelearn.testing import SAMPLES
    from royalelearn.testing import observations as build

    return build(mock_env_spec, seed=3, steps=SAMPLES)


@pytest.fixture
def rect(env_spec: Any, observations: list[dict[str, Any]]) -> Iterator[Any]:
    from royalelearn.testing import RectFixture

    built = RectFixture(env_spec, observations)
    built.fill()
    try:
        yield built
    finally:
        built.close()
