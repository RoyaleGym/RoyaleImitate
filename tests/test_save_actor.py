"""``save_actor``: a trained ``Learner`` written as an actor artifact a new run can start from.

On MockEngine and a step budget a test can afford: the folder holds the learner's weights and
probe rows, a student started from it reproduces the teacher's recorded log-probabilities
exactly, and ``examples/minimal.py``'s teacher-then-student flow runs (its own ``build_env`` is
swapped for MockEngine's; the real engine is the fresh-user test's).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from royaleimitate import save_actor
from royaleimitate.artifacts import artifact_digest, read_actor_artifact
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
        MockEngine(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(6),
    )


def test_a_learner_saved_as_an_actor_starts_a_student_that_plays_the_same(
    tmp_path: Path,
) -> None:
    teacher = Learner(build_env, n_envs=2, save_dir=tmp_path / "teacher", **TINY)
    with pytest.raises(PreflightError, match="learn"):
        save_actor(teacher, tmp_path / "early")
    teacher.learn(total_steps=32)
    folder = tmp_path / "teacher-actor"
    digest = save_actor(teacher, folder)
    assert digest == artifact_digest(folder)
    artifact = read_actor_artifact(folder)
    assert artifact.probe is not None and artifact.probe.rows.shape[0] > 0
    live = teacher.run.model.actor.state_dict()
    assert all(torch.equal(artifact.state[k], live[k].float()) for k in live)
    env = build_env()
    obs, _ = env.reset(seed=5)
    saved_bot = Learner.load_policy(folder, greedy=True)
    assert 0 <= saved_bot(obs["blue"]) < len(obs["blue"]["action_mask"])
    rebuilt = Learner.load_env(folder)
    try:
        assert (folder / "environment.json").is_file()
        obs, _ = rebuilt.reset(seed=5)
        assert 0 <= saved_bot(obs["blue"]) < len(obs["blue"]["action_mask"])
    finally:
        rebuilt.close()

    init = {"path": str(folder), "sha256": digest}
    student = Learner(
        build_env,
        n_envs=2,
        save_dir=tmp_path / "student",
        extensions={"warm_start": {"init": init}},
        **TINY,
    )
    student.learn(total_steps=32)
    assert student.run.extension_facts["warm_start"]["self_test"] == 0.0


def test_the_minimal_examples_flow_runs_on_mock_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "minimal.py"
    spec = importlib.util.spec_from_file_location("minimal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "build_env", build_env)
    # The example says no device, so it takes a GPU when there is one; this one says there is
    # none, which is CI's case and the CPU fallback's.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    student = module.main(steps=32)
    assert student.steps >= 32
    assert os.path.isdir(tmp_path / "runs" / "teacher-actor")
