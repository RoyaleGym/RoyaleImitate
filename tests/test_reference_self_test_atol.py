"""``imitation.references.<name>.self_test_atol``: a snapshot reference's self-test tolerance.

The properties, each held by a test that has been seen failing on a plant:

- unset, a reference whose log-probabilities on its probe rows moved by more than 1e-5 is refused,
  as it always was;
- set, the same reference loads, and a reference that moved past the stated tolerance is still
  refused;
- unset, the reference encodes exactly as before, so no run identity moves;
- a negative tolerance is a config problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from imitation_support import seeded_artifact, with_imitation
from royaleimitate.config import SnapshotReferenceSpec
from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.testing import coordinator, tiny_config

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

REGULARISER = {
    "kind": "reference_kl",
    "name": "bc",
    "reference": "bc",
    "budget": {"kind": "constant", "value": 0.1},
    "coef": {"start": 1.0},
}


def _nudged(tmp_path: Path, delta: float) -> tuple[Path, str]:
    """A reference artifact whose first weight moved by ``delta`` after its probe was recorded.

    On the tiny config, 0.01 moves the probe log-probabilities by ~5e-5 and 0.5 by ~2e-3.
    """

    def edit(tensors: dict[str, Any]) -> None:
        name = next(n for n, t in tensors.items() if t.is_floating_point() and t.numel() > 1)
        tensors[name].view(-1)[0] += delta

    folder = tmp_path / f"nudged-{delta}"
    return folder, seeded_artifact(
        tiny_config(tmp_path / f"donor-{delta}"), folder, coordinator, edit=edit
    )


def _config(tmp_path: Path, folder: Path, digest: str, **reference: Any) -> cfg.RunConfig:
    block = {"kind": "snapshot", "path": str(folder), "sha256": digest, **reference}
    return with_imitation(
        tiny_config(tmp_path / "run"), references={"bc": block}, regularisers=[REGULARISER]
    )


def test_unset_a_reference_that_moved_past_1e_5_is_refused(tmp_path: Path) -> None:
    folder, digest = _nudged(tmp_path, 0.01)
    with (
        pytest.raises(
            PreflightError, match=r"imitation\.references\.bc: the probe-logit self-test failed"
        ),
        coordinator(_config(tmp_path, folder, digest)),
    ):
        pass


def test_a_stated_tolerance_loads_it_and_still_refuses_a_larger_move(tmp_path: Path) -> None:
    """Plant: references.py back on its fixed 1e-5 makes the first start refuse."""
    folder, digest = _nudged(tmp_path, 0.01)
    with coordinator(_config(tmp_path, folder, digest, self_test_atol=1e-4)) as run:
        run.iterate()
    folder, digest = _nudged(tmp_path, 0.5)
    with (
        pytest.raises(PreflightError, match=r"against a tolerance of 0\.0001"),
        coordinator(_config(tmp_path, folder, digest, self_test_atol=1e-4)),
    ):
        pass


def test_unset_the_reference_encodes_as_before() -> None:
    """Plant: a plain float default (1e-5) puts the field into every reference's encoding."""
    unset = msgspec.json.encode(SnapshotReferenceSpec(path="a/bc", sha256="1" * 64))
    assert unset == b'{"kind":"snapshot","path":"a/bc","sha256":"' + b"1" * 64 + b'"}'
    stated = msgspec.json.encode(
        SnapshotReferenceSpec(path="a/bc", sha256="1" * 64, self_test_atol=0.02)
    )
    assert b'"self_test_atol":0.02' in stated


def test_a_negative_tolerance_is_refused_by_name(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "a", "1" * 64, self_test_atol=-1.0)
    problems = cfg.check_consistency(config)
    assert any(
        "imitation.references.bc.self_test_atol must be at least 0" in p for p in problems
    ), problems
