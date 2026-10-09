"""Write a saved bot and a clone with the RELEASED royalelearn 0.5.14 and royaleimitate 0.2.10.

Run from this folder's own venv (pip install royalelearn==0.5.14 royaleimitate==0.2.10), with
ROYALESIM_DATA_DIR pointing at a RoyaleSim checkout's data/ for MockEngine's 2018 tables.
"""

import sys
from pathlib import Path

import numpy as np

from royalegym import ClashParallelEnv, MockEngine
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royaleimitate import clone, record
from royalelearn import Learner
from royalelearn.testing import PREFLIGHT


def build_env():
    return ClashParallelEnv(
        MockEngine(), termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(6)
    )


def first_legal_play(obs):
    legal = np.flatnonzero(np.asarray(obs["action_mask"]))
    plays = legal[legal != 0]
    return int(plays[0]) if plays.size else 0


if __name__ == "__main__":
    out = Path(sys.argv[1])
    small = {
        "device": "cpu",
        "steps_per_update": 16,
        "trunk_channels": 8,
        "trunk_blocks": 1,
        "critic_hidden": 8,
        "seed": 1,
        "_coordinator_kwargs": {"preflight_kwargs": PREFLIGHT},
    }
    teacher = Learner(build_env, n_envs=2, save_dir=out / "run", **small)
    teacher.learn(total_steps=16)
    teacher.save(out / "bot-saved-by-royalelearn-0.5.14")
    demos = record(teacher, first_legal_play, out / "demos", battles=40, seed=1)
    clone(teacher, demos, out / "clone-by-royaleimitate-0.2.10", epochs=2, batch_size=64)
    print("done")
