# RoyaleImitate

[![suite](https://github.com/RoyaleGym/RoyaleImitate/actions/workflows/suite.yml/badge.svg)](https://github.com/RoyaleGym/RoyaleImitate/actions/workflows/suite.yml)

Start a Clash Royale bot from one you already have, and keep it close to that bot while it learns.
It is an optional add-on to RoyaleLearn, the trainer: leave it out and nothing changes.

## Install

    pip install "royalegym[all]"

This is the `[imitate]` part. Until it is on PyPI, see the [guide's Install](docs/guide.md#install).

## Try it

```python
from royalegym import TowerHPReward, make_env
from royalelearn import Learner
from royaleimitate import save_actor

def build_env():
    return make_env(reward=TowerHPReward())

teacher = Learner(build_env, save_dir="runs/teacher")
teacher.learn(total_steps=20_000)
digest = save_actor(teacher, "runs/teacher-actor")

start = {"path": "runs/teacher-actor", "sha256": digest}
student = Learner(build_env, save_dir="runs/student", extensions={"warm_start": {"init": start}})
student.learn(total_steps=20_000)
```

The student starts from the teacher's weights; [examples/minimal.py](examples/minimal.py) also keeps it close.

## Next

- The guide: [docs/guide.md](docs/guide.md). The full contract (Advanced): [docs/spec.md](docs/spec.md)
- Questions: [Discord](https://discord.gg/4D2BS5JBHP)

MIT licensed. See [LICENSE](LICENSE).
