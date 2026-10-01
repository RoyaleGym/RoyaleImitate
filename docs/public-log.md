# PublicLogMemory: the observation's fair fields from a timed log of card plays

`royaleimitate.public_log.PublicLogMemory` gives you one seat's fair observation fields with no
engine running. You give it a timed log of card plays and the seat's dealt deck order. It gives you
the numbers the env would show that seat at the same tick. Log the ability presses too: see
[Ability presses](#ability-presses).

## What it fills

Both elixir bars, your own hand and cycle, the cards the opponent has shown, and the time since the
last play. `fields()` lists the names in vector order: RoyaleGym's `FAIR_FIELDS`, plus
`enemy_last_card` when you pass `enemy_last_card=True`.

## What it cannot fill

The board: tower hitpoints, crowns and which kings are awake (`BOARD_FIELDS`). A log of plays does
not say what the plays did.

## Ability presses

A press of an ability button (a champion's, a hero's) plays no card, and it is paid from the bar: 1
elixir for a Golden Knight's, 3 for a hero Musketeer's, 2 for a hero Ice Golem's. Log each one with
`own_press(tick, elixir)` or `enemy_press(tick, elixir)`, dated like a play. Both counted bars then
charge it, as the env's `MatchMemory` does from RoyaleGym 276c3e9 on. The env reads a press off the
public ability rows.

A press missing from the log leaves that side's counted bar high by its cost, until the bar is full
again. A log with presses needs RoyaleGym 276c3e9 or later. A log without them runs on older ones as
before.

The env makes presses only when RoyaleGym's action parser has its opt-in ability buttons on. A log
of real matches has one wherever a champion's or hero's ability was used. A battle in
`tests/test_public_log.py` in which both seats press a Golden Knight checks every field against the
env at every step.

## The command delay

With a command delay (RoyaleGym's `command_delay_ticks`, on an engine that has one), a play or a
press is accepted on one tick and runs some ticks later. It is paid, and the hand moves, only when
it runs, as in the client. Log each one at the tick it runs.

A seat knows its own taps, so give its own commands the tick they were accepted too: `own_play(tick,
card, accepted=...)` and `own_press(tick, elixir, accepted=...)`. From the tick after that through
the tick it runs, the fields show the command waiting, as the env's do: the waiting card is flagged
in `own_hand_pending`, `own_pending_cost` shows what it holds, and the hand is priced from the bar
less that. A waiting Mirror holds the price it had when accepted. The opponent's waiting commands
are never an input.

A log without accepted ticks shows nothing waiting, and runs on a RoyaleGym before 5565645 as
before. Battles in `tests/test_public_log.py` with the two seats' commands waiting 21 and 22 ticks
check every field against the env at every step, with and without the Mirror in both decks.

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
- `own_press` and `enemy_press` log ability presses the same way: see
  [Ability presses](#ability-presses).
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
class adds only where the inputs come from: plays from the log, the own hand from the deck order,
and what each hand slot costs (`hand_costs()`), which for a Mirror is more than its listed elixir.

## The assumptions, and where each is checked

1. A played card's hand slot takes the next card, and the played card goes to the back of the queue.
2. A play at tick p is paid before tick p runs, and shows in any observation after tick p.
3. A match still running at the end of regulation is in overtime.
4. The elixir rate at a tick is `ElixirLaw.rate_at`.
5. A Mirror in the hand costs the card it would copy plus one. The card it copies is its side's
   last play that was not a Mirror. This one is checked against the engine's own price for each
   hand slot (`PlayerState.hand_costs`), in battles where both seats hold the Mirror.

The elixir counts charge a Mirror play the same price, the card it copied plus one. That count is
RoyaleGym's `MatchMemory`, which does it from RoyaleGym c6a36b0 on; before that it charged one
elixir, and a counted bar read too high after every Mirror play. The env writes a Mirror slot at
its price from RoyaleGym 74c3852 on. So the fields here match the env's with RoyaleGym 74c3852 or
later, as long as it can load your engine: a RoyaleGym older than its engine refuses to build a
RustEngine. On RoyaleSim 1e6a6a5 the oldest RoyaleGym that loads it and passes these tests is
600bfc8. A battle in `tests/test_public_log.py` with the Mirror in both decks checks every field
against the env at every step.

`tests/test_public_log.py` checks each one against MockEngine and RustEngine. It plays battles on
both, logs every accepted play, and compares every field with the env's vector at every step, for
both seats. The field names and their order must match too. The battles cover a full turn of both
queues, the switch to 2x and the tick regulation ends. A log dated one tick late must fail. The
RustEngine rows skip, with the reason, when the RoyaleSim core cannot be imported.
