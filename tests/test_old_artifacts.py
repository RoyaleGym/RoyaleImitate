"""Folders written by released versions from before ``actor_digest``.

``tests/data/old-artifacts`` holds a bot saved by royalelearn 0.5.14 and a clone made by
royaleimitate 0.2.10, written by ``make.py`` beside them with those versions installed from
PyPI. They load and warm-start as they always did. A run that differs from the clone only where
its actor's weights do not -- another critic -- is refused with the stamp named, and starts from
the clone once it is stamped. A run with another actor is refused as before.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from royaleimitate.artifacts import artifact_digest
from royaleimitate.stamp import stamp_actor_digest
from royalelearn import Learner
from royalelearn.errors import PreflightError
from royalelearn.testing import PREFLIGHT

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

OLD = Path(__file__).parent / "data" / "old-artifacts"
BOT = OLD / "bot-saved-by-royalelearn-0.5.14"
CLONE = OLD / "clone-by-royaleimitate-0.2.10"

#: The warm start's own self-test tolerance (``warm_start.init.self_test_atol``). The probe's
#: log-probabilities were recorded by the torch build ``make.py`` ran under, and another build
#: may differ in the last bits; the self-test is held to its tolerance, not to equality.
SELF_TEST_ATOL = 1e-5

#: The settings ``make.py`` trained with.
SMALL: dict[str, Any] = {
    "device": "cpu",
    "steps_per_update": 16,
    "trunk_channels": 8,
    "trunk_blocks": 1,
    "critic_hidden": 8,
    "_coordinator_kwargs": {"preflight_kwargs": PREFLIGHT},
}


def build_env():
    from royalegym import ClashParallelEnv, MockEngine
    from royalegym.done_condition import GameOverCondition, StepLimitCondition

    return ClashParallelEnv(
        MockEngine(), termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(6)
    )


def _copy(tmp_path: Path) -> tuple[Path, str]:
    """The clone, copied so a stamp never touches the checked-in folder, and its sha256."""
    folder = tmp_path / "clone"
    shutil.copytree(CLONE, folder)
    return folder, artifact_digest(folder)


def _student(tmp_path: Path, folder: Path, sha256: str, **settings: Any) -> Learner:
    return Learner(
        build_env,
        n_envs=2,
        save_dir=tmp_path / f"student-{sha256[:8]}",
        extensions={"warm_start": {"init": {"path": str(folder), "sha256": sha256}}},
        **{**SMALL, **settings},
    )


def test_the_clone_is_what_0_2_10_wrote() -> None:
    spec = json.loads((CLONE / "spec.json").read_text(encoding="utf-8"))
    assert "actor_digest" not in spec and spec["arch_digest"]


def test_an_old_saved_bot_loads_plays_and_rebuilds_its_environment() -> None:
    bot = Learner.load_policy(BOT, greedy=True)
    env = Learner.load_env(BOT)
    try:
        obs, _ = env.reset(seed=3)
        assert bool(np.asarray(obs["blue"]["action_mask"])[bot(obs["blue"])])
    finally:
        env.close()


def test_an_old_clone_warm_starts_a_run_of_its_own_architecture(tmp_path: Path) -> None:
    folder, sha256 = _copy(tmp_path)
    student = _student(tmp_path, folder, sha256)
    student.learn(total_steps=16)
    assert student.run.extension_facts["warm_start"]["self_test"] <= SELF_TEST_ATOL


def test_another_critic_names_the_stamp_and_starts_once_it_is_stamped(tmp_path: Path) -> None:
    """Plant: drop the check that names the stamp, and the refusal says only arch_digest."""
    folder, sha256 = _copy(tmp_path)
    with pytest.raises(PreflightError, match=r"python -m royaleimitate\.stamp"):
        _student(tmp_path, folder, sha256, critic_hidden=16).learn(total_steps=16)
    stamped = stamp_actor_digest(folder)
    assert stamped.old_sha256 == sha256 and stamped.new_sha256 != sha256
    student = _student(tmp_path, folder, stamped.new_sha256, critic_hidden=16)
    student.learn(total_steps=16)
    assert student.run.extension_facts["warm_start"]["self_test"] <= SELF_TEST_ATOL


def test_another_actor_is_refused_as_before_and_no_stamp_is_offered(tmp_path: Path) -> None:
    """Stamping would not help a run whose actor is another network, so it is not named."""
    folder, sha256 = _copy(tmp_path)
    with pytest.raises(PreflightError) as refused:
        _student(tmp_path, folder, sha256, trunk_channels=16).learn(total_steps=16)
    assert "arch_digest" in str(refused.value)
    assert "royaleimitate.stamp" not in str(refused.value)
