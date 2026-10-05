# Changelog

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
