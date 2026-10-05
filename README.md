<p align="center"><img src="docs/media/logo.png" width="128" alt="The RoyaleImitate logo: a purple crown shield with two white cards on it, outlined in gold"></p><h1 align="center">RoyaleImitate</h1>

<p align="center"><a href="https://github.com/RoyaleGym/RoyaleImitate/actions/workflows/suite.yml"><img alt="CI" src="https://github.com/RoyaleGym/RoyaleImitate/actions/workflows/suite.yml/badge.svg"></a> <img alt="License" src="https://img.shields.io/github/license/RoyaleGym/RoyaleImitate?style=flat-square&color=555"> <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB?style=flat-square&logo=python&logoColor=white"> <a href="https://royalegym.github.io/RoyaleGym/"><img alt="Docs" src="https://img.shields.io/badge/docs-royalegym.github.io-8957e5?style=flat-square&logo=readthedocs&logoColor=white"></a> <a href="https://discord.gg/cvRu4nEGXY"><img alt="Discord" src="https://img.shields.io/discord/1551699576304705647?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2"></a> <img alt="Last commit" src="https://img.shields.io/github/last-commit/RoyaleGym/RoyaleImitate?style=flat-square&color=555"></p>

Start a Clash Royale bot from one you already have, and keep it close to that bot while it learns.
It is an optional add-on to RoyaleLearn, the trainer: leave it out and nothing changes.

## Install

    pip install "royalegym[all]"

This is the `[imitate]` part. On Windows with an NVIDIA card, install PyTorch first: [Install](https://royalegym.github.io/RoyaleGym/install/), step 4.

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

- The docs: [royalegym.github.io/RoyaleGym](https://royalegym.github.io/RoyaleGym/), and the [RoyaleImitate page](https://royalegym.github.io/RoyaleGym/resources/royaleimitate/)
- The guide: [guide](https://royalegym.github.io/RoyaleGym/repos/royaleimitate/guide/). The full contract (Advanced): [spec](https://royalegym.github.io/RoyaleGym/repos/royaleimitate/spec/)
- Questions: [Discord](https://discord.gg/cvRu4nEGXY)

MIT licensed. See [LICENSE](LICENSE).
