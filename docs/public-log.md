# PublicLogMemory: the observation's fair fields from a timed log of card plays

`royaleimitate.public_log.PublicLogMemory` gives you one seat's fair observation fields with no
engine running. You give it a timed log of card plays and the seat's dealt deck order. It gives you
the numbers the env would show that seat at the same tick.

## What it fills

Both elixir bars, your own hand and cycle, the cards the opponent has shown, and the time since the
last play. `fields()` lists the names in vector order: RoyaleGym's `FAIR_FIELDS`, plus
`enemy_last_card` when you pass `enemy_last_card=True`.

## What it cannot fill

The board: tower hitpoints, crowns and which kings are awake (`BOARD_FIELDS`). A log of plays does
not say what the plays did.

## Use

```python
from royalegym.mock_engine import MockEngine
from royaleimitate.public_log import PublicLogMemory

CARD_NAMES = ["Knight", "Archer", "Goblins", "Giant", "Musketeer",
              "Skeletons", "Minions", "Valkyrie", "Fireball", "Arrows"]
cards = MockEngine(card_names=CARD_NAMES).cards()

memory = PublicLogMemory(cards, list(range(8)), card_names=CARD_NAMES)
memory.own_play(168, 0)    # the Knight in the first hand slot
memory.enemy_play(170, 9)  # the opponent plays Arrows
fields = memory.observe(200)
print(fields["own_elixir"], fields["enemy_cards_seen"])
```

- `cards` is the run's catalogue, in the run's order. Card ids are positions in it.
- The deck order is the seat's eight card ids as the match dealt them: the first four are the hand,
  slot by slot, then the queue, next card first.
- `start_tick`, `own_elixir_milli` and `enemy_elixir_milli` start the memory mid-match.
- `calibration` sets the elixir rules and the match clock. Pass the one your run's engine uses, or
  leave it out for the default.
- Feed plays in any amount ahead of time. `observe(tick)` sees exactly the plays made before that
  tick. Ticks only move forward, and a play dated before an observed tick is refused.
- `own_ticks_since_play` counts from the observation that first showed the play, as the env does.
- `unaffordable` counts plays the counted bar could not pay, as (own, enemy). It stays (0, 0) on a
  log an engine produced. Anything else means a play is missing from the log or the elixir law is
  different, and from then on the elixir fields are estimates.

## The `card_names` pin

Card ids are positions in the catalogue, so making one more card loadable renumbers every later id.
`card_names` is required. The names in `cards` must equal it exactly, in order, or the constructor
raises `ValueError`. The message names the first id that differs, the pinned name and the loaded
name there, and both lengths. Any difference refuses, including one extra card at the end.

Pass the list your run pins, the one its config names. Do not read it back from the catalogue: the
check would then compare the catalogue with itself and could never fail.

## One set of formulas

It is a thin wrapper over RoyaleGym's `MatchMemory` (`bind`, `start`, `advance`, `show_own_hand`)
and `fair_fields`. The env keeps its own `MatchMemory` with the same calls and reads its vector with
the same `fair_fields`. Of the calls, only the clock differs: the env reads it from the engine, and
this class works it out from the tick with `MatchClock.at`. Apart from the `card_names` check, this
class adds only where the inputs come from: plays from the log, and the own hand from the deck
order.

## The assumptions, and where each is checked

1. A played card's hand slot takes the next card, and the played card goes to the back of the queue.
2. A play at tick p is paid before tick p runs, and shows in any observation after tick p.
3. A match still running at the end of regulation is in overtime.
4. The elixir rate at a tick is `ElixirLaw.rate_at`.

`tests/test_public_log.py` checks each one against MockEngine and RustEngine. It plays battles on
both, logs every accepted play, and compares every field with the env's vector at every step, for
both seats. The field names and their order must match too. The battles cover a full turn of both
queues, the switch to 2x and the tick regulation ends. A log dated one tick late must fail. The
RustEngine rows skip, with the reason, when the RoyaleSim core cannot be imported.
