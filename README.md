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

And a `royaleimitate` command for the files those sections read:

- `royaleimitate artifact-digest <folder>` prints the digest a config names a folder by.
- `royaleimitate fit-field-reference` fits the small play-or-wait model a `field_mlp` reference
  loads.

And one class that needs no engine:

- `royaleimitate.public_log.PublicLogMemory` rebuilds the observation's fair fields with no engine
  running. You give it a timed log of card plays and the seat's dealt deck order. It gives you both
  elixir bars, your own hand and cycle, the cards the opponent has shown and the time since the
  last play. It needs the `card_names` list your run pins. [docs/public-log.md](docs/public-log.md)
  explains it.

## Install

Install RoyaleLearn first, following its README. That leaves you in a `Royale` folder with the
four public repos side by side and one virtual environment they share.

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

The run's identity records each section by content and this package by version and commit. Its
folder is checked for uncommitted edits like RoyaleLearn's own. [docs/spec.md](docs/spec.md) is the
full contract, and [docs/alarms.md](docs/alarms.md) lists the four alarms these sections add.

## Tests

```bash
python -m pytest -q
```

The tests need RoyaleLearn and RoyaleGym installed. They do not need this package installed: the
test session writes its install metadata for this checkout.
