"""Installed from a wheel, as ``pip install "royalegym[all]"`` installs it, the package works.

Every other test here runs against this checkout, and a checkout has a commit to name. A wheel in
site-packages has none, which is where a run using ``warm_start`` was refused for every user of
the release. So this builds a wheel from the checkout, installs it into a folder of its own,
and asks RoyaleLearn, in a process that imports that copy, to name the package in a run's
identity.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import sys
from pathlib import Path

import royaleimitate
from royalelearn import config as cfg
from royalelearn.extensions import with_sections
from royalelearn.identity import extension_records

site = Path(sys.argv[1]).resolve()
assert Path(royaleimitate.__file__).resolve().is_relative_to(site), royaleimitate.__file__
config = with_sections(
    cfg.RunConfig(env=cfg.default_env_spec(cfg.MOCK_ENGINE)),
    warm_start={"init": {"path": "artifacts/a", "sha256": "0" * 64}},
)
record = extension_records(config)["warm_start"]
print("GIT", record.git)
print("VERSION", record.version)
"""


@pytest.fixture(scope="module")
def wheel_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("wheel")
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "--no-build-isolation",
         "-w", str(work / "dist"), "-q"],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    (wheel,) = (work / "dist").glob("royaleimitate-*.whl")
    site = work / "site"
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site), str(wheel),
         "-q"],
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr[-2000:]
    return site


def test_a_wheel_installed_package_is_named_in_a_runs_identity(wheel_site: Path) -> None:
    """Plant: refuse an unknown commit again, as RoyaleLearn before 0.5.2 did, and this fails
    with the refusal every release user met."""
    import royalelearn

    learn_root = str(Path(royalelearn.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(wheel_site), learn_root])}
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(wheel_site)],
        capture_output=True,
        text=True,
        env=env,
        cwd=wheel_site.parent,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = dict(line.split(" ", 1) for line in done.stdout.splitlines() if " " in line)
    assert lines["GIT"].startswith("content:"), done.stdout
    import royaleimitate

    assert lines["VERSION"] == royaleimitate.__version__
