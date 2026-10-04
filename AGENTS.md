# AGENTS.md

Notes for AI coding agents working in this repo. People should start at the [README](README.md).

## What this is

RoyaleImitate is an optional add-on to RoyaleLearn: start a bot from saved weights, keep it near
a reference policy while it learns, and turn games into training rows and clones. RoyaleLearn
finds its two config sections, `warm_start` and `imitation`, through entry points.

## Layout

| Path | What it holds |
|---|---|
| `royaleimitate/cloning.py` | `record` (a teacher's games as rows) and `clone` (behaviour cloning) |
| `royaleimitate/replays.py` | `from_replays`: human games from the IL_Replay dataset as rows |
| `royaleimitate/artifacts.py` | Saved actor folders, their probe rows and `save_actor` |
| `royaleimitate/shards.py`, `split.py` | Demonstration rows on disk, and the by-match validation split |
| `royaleimitate/warm_start.py`, `init.py` | The `warm_start` section |
| `royaleimitate/extension.py`, `regularisers.py`, `references.py` | The `imitation` section |
| `royaleimitate/public_log.py` | The fair observation fields from a timed log of card plays |
| `docs/guide.md`, `docs/spec.md` | The guide, and the full contract |

## Build and test

Build RoyaleLearn from source first, then install this repo into the same virtual environment.
CI's own commands, from this folder:

    python -m pytest -q -p no:randomly -rs
    python -m ruff check royaleimitate tests

## Rules that tests enforce

- Only `royalelearn.extensions` is imported from RoyaleLearn. Anything else may change.
- Shard rows are stored exactly and refused when read by a run with another engine or
  observation.
- A wheel install works: a test builds the wheel and installs it.

## Public text

This repo is public. Describe the library and its defaults. Never add training results,
recipes, tuned settings, run names, or paths and names from private repos or machines.
