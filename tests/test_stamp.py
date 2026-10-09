"""``royaleimitate.stamp``: ``actor_digest`` added to a folder written before it existed.

A student whose architecture differs from its clone's only where the actor's weights do not feel
it -- here the critic's hidden width -- is refused by a folder held to the whole ``arch_digest``,
and loads it once the folder states what its actor computes. Stamping rewrites ``spec.json``
alone: every tensor stays byte for byte, and the folder's sha256 moves to a printed new value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from royaleimitate import save_actor
from royaleimitate.artifacts import artifact_digest
from royaleimitate.stamp import main, stamp_actor_digest
from royalelearn import Learner
from royalelearn.errors import PreflightError
from royalelearn.testing import PREFLIGHT

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

TINY = {
    "device": "cpu",
    "steps_per_update": 16,
    "_coordinator_kwargs": {"preflight_kwargs": PREFLIGHT},
}


def build_env():
    from royalegym import ClashParallelEnv, MockEngine
    from royalegym.done_condition import GameOverCondition, StepLimitCondition

    return ClashParallelEnv(
        MockEngine(), termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(6)
    )


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stamp")
    teacher = Learner(build_env, n_envs=2, save_dir=root / "teacher", **TINY)
    teacher.learn(total_steps=16)
    folder = root / "actor"
    save_actor(teacher, folder)
    return folder


def _unstamped(saved: Path, where: Path) -> Path:
    """The folder as one written before ``actor_digest`` existed."""
    import shutil

    folder = where / "old"
    shutil.copytree(saved, folder)
    spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
    assert spec["actor_digest"]
    del spec["actor_digest"]
    (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return folder


def _student(where: Path, folder: Path, sha256: str) -> Learner:
    return Learner(
        build_env,
        n_envs=2,
        save_dir=where / f"student-{sha256[:8]}",
        critic_hidden=32,
        extensions={"warm_start": {"init": {"path": str(folder), "sha256": sha256}}},
        **TINY,
    )


def test_a_stamped_folder_starts_a_student_with_another_critic(
    saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plant: stamp without writing the field, and the student is refused as before."""
    folder = _unstamped(saved, tmp_path)
    old = artifact_digest(folder)
    with pytest.raises(PreflightError, match="arch_digest"):
        _student(tmp_path, folder, old).learn(total_steps=16)

    weights = (folder / "actor.safetensors").read_bytes()
    probe = (folder / "probe.safetensors").read_bytes()
    assert main([str(folder)]) == 0
    new = artifact_digest(folder)
    said = capsys.readouterr().out
    assert old in said and new in said and new != old and "stamped" in said
    assert (folder / "actor.safetensors").read_bytes() == weights
    assert (folder / "probe.safetensors").read_bytes() == probe

    student = _student(tmp_path, folder, new)
    student.learn(total_steps=16)
    assert student.run.extension_facts["warm_start"]["self_test"] == 0.0


def test_a_folder_written_now_already_states_it_and_is_left_alone(saved: Path) -> None:
    before = artifact_digest(saved)
    done = stamp_actor_digest(saved)
    assert done.old_sha256 == done.new_sha256 == before


def test_a_policy_record_that_describes_other_weights_is_refused(
    saved: Path, tmp_path: Path
) -> None:
    """The digest is read from policy.json, so that file must be shown to be these weights'.
    Plant: skip the check, and a folder is stamped with another network's digest."""
    folder = _unstamped(saved, tmp_path)
    record = json.loads((folder / "policy.json").read_text(encoding="utf-8"))
    record["net"]["noop_bias"] = 3.0
    (folder / "policy.json").write_text(json.dumps(record), encoding="utf-8")
    before = (folder / "spec.json").read_bytes()
    with pytest.raises(PreflightError, match="describes arch_digest"):
        stamp_actor_digest(folder)
    assert (folder / "spec.json").read_bytes() == before
