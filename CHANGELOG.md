# Changelog

## Unreleased

- `ShardReader` keeps each part's packed rows after their first read, so a second epoch reads
  no part file again: `cache="auto"` (the default) keeps them in memory while every split's
  packed rows fit `cache_memory_mb` (2 GB), else as uncompressed `.npy` files in the shard's
  `packed` folder while the disk has room, which a later run on the same rows and codec reads
  too, else not at all. `"memory"`, `"disk"` and None choose. The rows and their order are
  the same whichever it is. `clone(cache=...)` passes it on and says which in its second line.

## 0.2.12

- `python -m royaleimitate.stamp FOLDER` (`royaleimitate.stamp.stamp_actor_digest`) adds
  `actor_digest` to an actor folder written before it existed, from the folder's own
  `policy.json` once that is shown to give the `arch_digest` the weights were saved with.
  Only `spec.json` changes: every tensor stays byte for byte, and the folder's old and new
  sha256 are printed. A stamped folder warm-starts a run that differs from it only in the
  precision, the device, the initialisation or the critic. Needs royalelearn 0.5.16.
- A warm start or a reference from a folder written before `actor_digest`, into a run that
  differs from it only there, is refused with the stamp command named; folders written by
  royaleimitate 0.2.10 and royalelearn 0.5.14 load and warm-start as before (tested on real
  ones, `tests/data/old-artifacts`).

## 0.2.11

Tagged, never published to PyPI: its CI failed on the tag. Everything below is in 0.2.12.

- `clone` says what it is doing: a first line with the rows, the device and the most epochs it
  may run, then one line per epoch with its validation NLL, the best so far, how long it took
  and at most how much longer the rest could take at that pace, and a line when it stops early.
  `printer=` takes them like `from_replays`; None prints nothing.

## 0.2.10

- `clone` and `save_actor` write `environment.json` beside the bot, what the learner's
  `build_env` built, so that `Learner.load_env(folder)` rebuilds the environment it plays in from
  the folder alone. Needs royalelearn 0.5.12.

## 0.2.9

- Demonstration rows store RoyaleGym's `unit_ids` observation as a column of its own, and a
  reading run with another unit-plane count is refused.

## 0.2.8

- `from_replays(..., keep=...)` uses only the matches a function you pass accepts, given each
  match's decks and plays; the others are skipped and counted as "not kept".

## 0.2.7

- `imitation.references.<name>.self_test_atol` sets a snapshot reference's probe self-test
  tolerance, as `warm_start.init.self_test_atol` does for the init. Unset, it stays 1e-5 and the
  reference encodes as before.

## 0.2.6

- Demonstration rows store RoyaleGym's `spell_ids` observation beside `card_ids`. Needs
  royalelearn 0.5.4.

## 0.2.5

- `from_replays(learner, out)` turns human games from the public IL_Replay dataset into rows
  that `clone` reads. Install with `pip install "royaleimitate[replays]"`.

## 0.2.4

- `record` takes a scripted teacher's name with hyphens or underscores.

## 0.2.3

- Works when installed from a wheel. Needs royalelearn 0.5.2.

## 0.2.2

- `PublicLogMemory` follows the hand refill timer: a played slot stays empty until it refills.

## 0.2.1

- A clone and a saved actor load as bots with `Learner.load_policy`.

## 0.2.0

- `record` plays a teacher in your environment and stores its decisions; `clone` trains a
  network to copy them. Needs royalelearn 0.5.0.

## 0.1.0

- `save_actor`, the `warm_start` and `imitation` sections, and a minimal example.
