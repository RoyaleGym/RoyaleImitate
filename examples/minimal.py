"""Start a new bot from a trained one, and keep it close to that bot while it learns.

    python minimal.py

It trains a first bot (the teacher) on your CPU, saves it, then starts a second bot (the
student) from the teacher's weights. While the student learns, a penalty keeps its moves
close to the teacher's.
"""

from royalegym import TowerHPReward, make_env
from royaleimitate import save_actor
from royalelearn import Learner


def build_env():
    return make_env(reward=TowerHPReward())


def main(steps: int = 20_000) -> Learner:
    teacher = Learner(build_env, n_envs=4, save_dir="runs/teacher")
    teacher.learn(total_steps=steps)
    digest = save_actor(teacher, "runs/teacher-actor")

    teacher_actor = {"path": "runs/teacher-actor", "sha256": digest}
    student = Learner(
        build_env,
        n_envs=4,
        save_dir="runs/student",
        extensions={
            # Start from the teacher's weights.
            "warm_start": {"init": teacher_actor},
            # Keep the student's moves close to the teacher's.
            "imitation": {
                "references": {"teacher": {"kind": "snapshot", **teacher_actor}},
                "regularisers": [
                    {
                        "kind": "reference_kl",
                        "name": "teacher",
                        "reference": "teacher",
                        "budget": {"kind": "constant", "value": 0.05},
                        "coef": {"start": 0.3},
                    }
                ],
            },
        },
    )
    student.learn(total_steps=steps)
    return student


if __name__ == "__main__":
    main()
