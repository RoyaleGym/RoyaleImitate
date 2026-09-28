# RoyaleImitate

Imitation learning for the Royale stack, as an optional add-on to RoyaleLearn. Install it next to
RoyaleLearn when you want a policy to start from saved weights or stay near a reference policy
while it learns. Leave it out and RoyaleLearn behaves as if it did not exist.

## What you get

Two config sections for a RoyaleLearn run:

- `warm_start` starts the actor from a saved policy folder. It checks the loaded weights against
  the probe rows stored with them, and it can hold the actor still for the first iterations while
  the critic catches up.
- `imitation` keeps the policy near one or more reference policies. Each reference-KL
  regulariser has a budget and a coefficient that adapts to stay inside it.

And a `royaleimitate` command for the files those sections read. Run it as
`.venv\Scripts\royaleimitate` (`.venv/bin/royaleimitate` on macOS and Linux), or as
`.venv\Scripts\python -m royaleimitate`, unless the virtual environment is activated:

- `royaleimitate artifact-digest <folder>` prints the digest a config names a folder by.
- `royaleimitate fit-field-reference --rows <file.npz> --fields a,b,c --out <folder>` fits the
  small play-or-wait model a `field_mlp` reference loads. The rows file is an `.npz` with one column
  per named field (rows by that field's width in the observation vector), and three columns of one
  value a row: `label` (1 played, 0 waited), `weight` (0 or more) and `group` (rows of one group
  land on the same side of the validation split). Nothing in the public packages makes such rows
  from real games, and no demonstration rows are distributed with these repos.

And one class that needs no engine:

- `royaleimitate.public_log.PublicLogMemory` rebuilds the observation's fair fields with no engine
  running. You give it a timed log of card plays and the seat's dealt deck order. It gives you both
  elixir bars, your own hand and cycle, the cards the opponent has shown and the time since the
  last play. It needs the `card_names` list your run pins. [docs/public-log.md](docs/public-log.md)
  explains it.

## Install

Install RoyaleLearn first, following its README, including its torch extra: this package needs
torch and safetensors. That leaves you in a `Royale` folder with the public repos side by side
and one virtual environment they share.

These are Windows PowerShell commands, like RoyaleLearn's. On macOS or Linux, use
`.venv/bin/python` in place of `.venv\Scripts\python`. From the `Royale` folder:

```powershell
git clone https://github.com/RoyaleGym/RoyaleImitate.git
.venv\Scripts\python -m pip install -e RoyaleImitate --no-deps
```

RoyaleLearn finds the two sections through this package's entry points. A config that names
`warm_start` or `imitation` without this package installed is refused at load, naming the section.

## Use

Add the sections to a run's config. For example, to start from a saved policy and hold it still for
the first 80,000 environment steps:

```json
{
  "warm_start": {
    "init": {"path": "artifacts/my-policy", "sha256": "<digest from artifact-digest>"},
    "actor_lr_scale": {"kind": "piecewise", "points": [[0, 0.0], [80000, 1.0]]}
  }
}
```

No command in the public packages writes a saved policy folder yet. `royalelearn bc` and
`royalelearn export`, which would, are specified in [docs/spec.md](docs/spec.md) and not built.
Until they are, write one from Python with `royaleimitate.artifacts.write_actor_artifact`, the
way `write_from_run` in `tests/imitation_support.py` does: it stores the actor's weights and the
probe rows the load is checked against. A RoyaleLearn pool snapshot is refused (it carries no
probe rows), and so is a checkpoint folder (it has no `spec.json`).

`royaleimitate artifact-digest <folder>` prints the digest to paste into `sha256`. `doctor` does
not open the folder or check the digest: `train` does, and refuses a missing folder or a wrong
digest. A relative `path` is read from the folder you run the command in, not from the config
file's folder.

The run's identity records each section by content and this package by version and commit. Its
folder is checked for uncommitted edits like RoyaleLearn's own. [docs/spec.md](docs/spec.md) is the
full contract, and [docs/alarms.md](docs/alarms.md) lists the four alarms these sections add.

## Tests

From the `Royale` folder, go into `RoyaleImitate` and run the tests there with the shared
environment's python (`../.venv/bin/python` on macOS and Linux):

```powershell
cd RoyaleImitate
..\.venv\Scripts\python -m pytest -q
```

On 2026-09-28 at 03f8e82 that printed `121 passed` on Windows with the engine built. Run from
the `Royale` folder without the `cd`, pytest collects the tests of every repo there and errors.

The tests need RoyaleLearn and RoyaleGym installed. They do not need this package installed: the
test session writes its install metadata for this checkout.
