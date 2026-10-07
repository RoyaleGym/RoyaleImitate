"""``record`` and ``clone``: a teacher's battles as demonstrations, and a policy trained to copy it.

On MockEngine and budgets a test can afford. ``record`` plays the teacher on both seats of battles
in a learner's own environment and stores every decision as a shard row, one group per battle.
``clone`` trains the learner's own network on those rows -- cross-entropy of the teacher's action
under the masked distribution, the by-battle validation split, AdamW and early stopping -- and
writes an actor folder a ``warm_start`` loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from royaleimitate import clone, record
from royaleimitate.artifacts import artifact_digest, read_actor_artifact
from royaleimitate.shards import ShardContext, ShardReader
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
        truncation_cond=StepLimitCondition(8),
    )


def first_legal_play(obs: dict[str, Any]) -> int:
    """A teacher a network can learn: the first legal play, else the no-op."""
    legal = np.flatnonzero(np.asarray(obs["action_mask"]))
    plays = legal[legal != 0]
    return int(plays[0]) if plays.size else 0


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Learner, Path]:
    root = tmp_path_factory.mktemp("clone")
    learner = Learner(build_env, n_envs=2, save_dir=root / "student", **TINY)
    directory = record(learner, first_legal_play, root / "demos", battles=80, seed=1)
    return learner, directory


def test_record_stores_every_decision_the_teacher_made(recorded: Any) -> None:
    learner, directory = recorded
    reader = ShardReader(directory, ShardContext.of_config(learner.config))
    assert reader.manifest.rows > 0
    assert reader.manifest.validation_rows > 0, "no battle landed in the validation split"
    columns = reader.part(0)
    observations = reader.observations(columns)
    assert [first_legal_play(o) for o in observations] == columns["action"].tolist()
    assert set(columns["seat"].tolist()) == {0, 1}
    assert len(set(columns["group"].tolist())) > 1


def test_a_teacher_is_a_policy_an_opponent_or_a_scripted_name(tmp_path: Path) -> None:
    learner = Learner(build_env, n_envs=2, save_dir=tmp_path / "s", **TINY)
    for teacher in ("random", "push", "first-affordable", "first_affordable"):
        assert record(learner, teacher, tmp_path / teacher, battles=2).is_dir()
    with pytest.raises(PreflightError, match="teacher"):
        record(learner, "nobody", tmp_path / "x", battles=1)


def test_clone_copies_the_teacher_and_a_student_starts_from_it(
    recorded: Any, tmp_path: Path
) -> None:
    learner, directory = recorded
    out = tmp_path / "clone"
    digest = clone(learner, directory, out, epochs=6, batch_size=64)
    assert digest == artifact_digest(out)
    artifact = read_actor_artifact(out)
    assert artifact.probe is not None and artifact.probe.rows.shape[0] > 0
    history = artifact.spec.meta["clone"]["validation_nll"]
    assert history[-1] < history[0], f"validation NLL did not fall: {history}"

    # The clone plays on its own, before any RL: as a bot, legal on a fresh battle.
    bot = Learner.load_policy(out, greedy=True)
    env = build_env()
    obs, _ = env.reset(seed=5)
    assert bool(np.asarray(obs["blue"]["action_mask"])[bot(obs["blue"])])

    # A clone that never learned has its environment beside it too, and it rebuilds from the
    # folder alone: no build_env, no script run.
    import msgspec

    rebuilt = Learner.load_env(out)
    try:
        assert msgspec.json.decode(msgspec.json.encode(rebuilt.config())) == learner.environment
        obs, _ = rebuilt.reset(seed=5)
        assert bool(np.asarray(obs["blue"]["action_mask"])[bot(obs["blue"])])
    finally:
        rebuilt.close()

    student = Learner(
        build_env,
        n_envs=2,
        save_dir=tmp_path / "student",
        extensions={"warm_start": {"init": {"path": str(out), "sha256": digest}}},
        **TINY,
    )
    student.learn(total_steps=16)
    assert student.run.extension_facts["warm_start"]["self_test"] == 0.0


def test_clone_refuses_demonstrations_with_no_validation_battle(tmp_path: Path) -> None:
    learner = Learner(build_env, n_envs=2, save_dir=tmp_path / "s", **TINY)
    few = record(learner, first_legal_play, tmp_path / "few", battles=1, seed=0)
    reader = ShardReader(few, ShardContext.of_config(learner.config))
    if reader.manifest.validation_rows:
        pytest.skip("this one battle happened to land in the validation split")
    with pytest.raises(PreflightError, match="record 100 battles or more"):
        clone(learner, few, tmp_path / "out", epochs=1)


def test_the_package_functions_stay_functions_after_use(recorded: Any, tmp_path: Path) -> None:
    import royaleimitate

    learner, _directory = recorded
    royaleimitate.record(learner, "noop", tmp_path / "a", battles=1)
    assert callable(royaleimitate.record) and callable(royaleimitate.clone)
    assert not isinstance(royaleimitate.clone, type(royaleimitate))
