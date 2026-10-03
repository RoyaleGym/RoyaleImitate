"""The fair observation fields a timed log of card plays determines, with no engine.

WHAT IT IS FOR. A log of who played which card when, plus the rules everyone knows,
fixes most of what a player remembers: both elixir bars, the own hand and cycle, the
cards the opponent has shown, how long since the last play. ``PublicLogMemory`` rebuilds
those fields from such a log, so a model can be trained on a timed log of card plays
with no engine running, and read the numbers the env would show. The log also takes the
ability presses, which are paid from the bar and play no card (below).

ONE SET OF FORMULAS. It is a thin wrapper over ``MatchMemory`` in ``royalegym.obs``:
``bind`` gives it the card costs, ``start`` starts it, and ``advance`` and
``show_own_hand`` move it on (the calls the env's ``observe`` makes). The fields are
read with ``fair_fields`` (the function ``build_vector`` gets them from), on
``MatchClock.at``. Apart from the catalogue pin, what this module adds is only where
the inputs come from: plays from a log, and the own hand from the dealt deck order.

WHAT IT CANNOT FILL. The board: tower hitpoints, crowns and which kings are awake
(``BOARD_FIELDS``). A log of plays does not say what the plays did. Nor the card status
fields (``SpatialObsBuilder(card_status=True)``, ``CARD_STATUS_FIELDS``): evolution
progress, hero and ability buttons, and the forms the enemy has shown come from the
engine's state and the units on the board, not from which card was played when.
``observe`` returns only ``fields()``, which holds none of them.

ABILITY PRESSES. A press of an ability button (a champion's, a hero's) plays no card and
is paid from the bar: 1 elixir for a Golden Knight's, 3 for a hero Musketeer's, 2 for a hero
Ice Golem's. Log each with ``own_press`` or ``enemy_press``, its tick and its elixir, dated
like a play, and both counts charge it, as the env's ``MatchMemory`` does from RoyaleGym
276c3e9 on (the env reads a press off the public ability rows). A press missing from the log
leaves that side's counted bar high by its cost until the bar is full again. A log with
presses needs RoyaleGym 276c3e9 or later; a log without them runs on older ones as before.
The env makes presses only when RoyaleGym's action parser has its opt-in ability buttons on;
a log of real matches has one wherever a champion's or hero's ability was used.

THE COMMAND DELAY. With a command delay (RoyaleGym's ``command_delay_ticks``, on an engine
that has one) a play or a press is accepted on one tick and runs, and is paid, some ticks
later; the hand and the bar move only when it runs, as the client's do. Log each at the tick
it RUNS. The seat knows its own taps, so give its own commands the tick they were accepted
too (``own_play(..., accepted=)``, ``own_press(..., accepted=)``): from the tick after that
through the tick it runs, the fields show the command waiting, as the env's do
(``own_hand_pending``, ``own_pending_cost``, and a hand priced from the bar less what
waits). A waiting play holds the price it had when accepted; a Mirror's is its copy's plus
one, the copy being the seat's last play that had run by then. The opponent's waiting
commands are never an input. A log without accepted ticks shows nothing waiting and runs on
RoyaleGym before 5565645 as before.

THE CATALOGUE PIN. Card ids are positions in the catalogue, so making one more card
loadable renumbers every later id. ``card_names`` pins the catalogue by name, and the
constructor refuses cards whose names differ from it in any way. Pass the list your run
pins (the one its config names), not one read back from the catalogue: reading it back
would make the check compare the catalogue with itself.

THE ASSUMPTIONS, EACH CHECKED AGAINST AN ENGINE IN tests/test_public_log.py:
    * a played card goes to the back of the queue and its hand slot empties; the slot
      waits for the side's refill timer (``match.HAND_REFILL_MS_1X`` / ``_2X`` / ``_3X``,
      by the elixir rate): an idle timer refills it at once from the front of the queue
      and restarts at its period, and a running one refills the lowest empty slot when
      it runs out (with no such keys in the calibration, a refill is instant);
    * a play at tick p is paid before tick p runs and shows in any observation at a
      tick after p;
    * a match still running at the end of regulation is in overtime;
    * the elixir rate at a tick is ``ElixirLaw.rate_at``;
    * a Mirror in the hand costs the listed elixir of its side's last play that was not
      a Mirror, plus one, and -1 before there is one (``hand_costs``);
    * an ability press at tick p is paid like a play at p, at its button's cost, and
      moves no hand slot;
    * under a command delay, an own command accepted at a and run at r waits in every
      observation at a tick in a+1..r, holding the price it had when accepted.

THE MIRROR. ``observe`` passes the hand's prices to ``fair_fields``, and the elixir counts
come from ``MatchMemory``, which charges a Mirror play its copy plus one from RoyaleGym
c6a36b0 on. With a RoyaleGym older than that it charged the listed one elixir, so after a
Mirror play the counted bar of the side that played it read too high by the copy's elixir.
The env writes the same prices from RoyaleGym 74c3852 on: the log's fields match the env's
only when both sides have them. A RoyaleGym must also be able to load the engine at all: one
older than its engine refuses to build a RustEngine. On RoyaleSim 1e6a6a5 the oldest that
does, and passes tests/test_public_log.py, is 600bfc8.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from royalegym.obs import BOARD_FIELDS, FAIR_FIELDS, MatchClock, MatchMemory, fair_fields
from royalegym.protocol import (
    DECK_SIZE,
    EMPTY_CARD,
    HAND_SIZE,
    Calibration,
    CardInfo,
    ElixirLaw,
    Placement,
    default_calibration,
)

__all__ = ["BOARD_FIELDS", "PublicLogMemory"]

_OWN, _ENEMY = 0, 1
#: ``PlayerState.pending`` row kinds.
_KIND_PLAY, _KIND_PRESS = 0, 1


class PublicLogMemory:
    """One seat's fair, non-board observation fields, kept from a timed log of plays.

    ``cards`` is the run's catalogue, in the run's order: ids are positions, and the
    one-hot fields are as wide as it is. ``deck_order`` is the seat's own eight cards
    as the match dealt them: the first four are the hand, slot by slot, and the rest
    the queue, next card first. ``card_names`` is the catalogue the run pins, by name,
    in id order. The names in ``cards`` must equal it exactly, or the constructor
    refuses before any id is read.

    Feed plays with ``own_play`` and ``enemy_play``, in any amount ahead of time, and
    read the fields with ``observe(tick)``. An observation at tick T sees exactly the
    plays made before T, as the env's does.

    ``refill_timer`` (default on, as in the client and RustEngine) makes a played card's slot
    wait for the refill timer; off, a slot refills at once, as in RoyaleGym's MockEngine.

    ``unaffordable`` counts plays the counted bar could not pay, (own, enemy). The
    engine refuses those, so on a log an engine produced it stays (0, 0). Anywhere else
    a non-zero count means a play is missing from the log or the elixir law is not the
    one this memory counts with, and from then on the elixir fields are estimates.
    """

    def __init__(
        self,
        cards: Sequence[CardInfo],
        deck_order: Sequence[int],
        *,
        card_names: Sequence[str],
        calibration: Calibration | None = None,
        start_tick: int = 0,
        own_elixir_milli: int | None = None,
        enemy_elixir_milli: int | None = None,
        enemy_last_card: bool = False,
        refill_timer: bool = True,
    ) -> None:
        self.cards = list(cards)
        self.card_names = list(card_names)
        self._check_card_names()
        n = len(self.cards)
        deck = [int(c) for c in deck_order]
        if len(deck) != DECK_SIZE or len(set(deck)) != DECK_SIZE:
            raise ValueError(f"deck_order must be {DECK_SIZE} distinct card ids, got {deck}")
        if not all(0 <= c < n for c in deck):
            raise ValueError(f"deck_order {deck} has ids outside a {n}-card catalogue")
        self.calibration = calibration
        cal = calibration if calibration is not None else default_calibration()
        self.law = ElixirLaw.load(cal)
        self.max_mana = cal.int("match.MAX_MANA")
        clock = MatchClock.at(start_tick, calibration)
        self.regular_ticks = clock.regular_ticks
        self.overtime_ticks = clock.overtime_ticks
        start = 1000 * cal.int("match.START_MANA")
        self.enemy_last_card = enemy_last_card
        self.hand = deck[:HAND_SIZE]
        self.queue = deck[HAND_SIZE:]
        #: The hand refill timer, ms left, and its period at the 1x, 2x and 3x rates.
        self.refill_ms = 0
        self._refill_periods = _refill_periods(cal) if refill_timer else (0, 0, 0)
        #: The first tick whose refill step has not run yet.
        self._refill_tick = int(start_tick)
        #: What a Mirror played now would copy: this side's last play that was not a Mirror.
        self.mirror_target: int | None = None
        self._pending: list[tuple[int, int, int, int]] = []  # (tick, order fed, side, card)
        self._presses: list[tuple[int, int, int, int]] = []  # (tick, order fed, side, elixir)
        #: This seat's commands under a delay: (accepted, runs, kind, what, cost or None).
        self._waiting: list[tuple[int, int, int, int, int | None]] = []
        #: This seat's plays that were not a Mirror, (the tick each ran, card), oldest first.
        self._ran: list[tuple[int, int]] = []
        self._fed = 0
        self.memory = MatchMemory(n, self.law)
        self.memory.bind(self.cards)
        self.memory.start(
            start_tick,
            start if own_elixir_milli is None else own_elixir_milli,
            start if enemy_elixir_milli is None else enemy_elixir_milli,
            self.hand,
            self.queue[0],
        )

    def _check_card_names(self) -> None:
        """Refuse a catalogue whose names are not exactly the ones pinned."""
        names = [c.name for c in self.cards]
        pinned = self.card_names
        if pinned != names:
            i = next(
                (k for k, (a, b) in enumerate(zip(pinned, names, strict=False)) if a != b),
                min(len(pinned), len(names)),
            )
            was = pinned[i] if i < len(pinned) else "<end>"
            now = names[i] if i < len(names) else "<end>"
            raise ValueError(
                f"card_names pins a catalogue that has {was!r} at id {i}; the cards given "
                f"have {now!r} there ({len(pinned)} pinned names, {len(names)} loaded). "
                "Catalogue ids are positional, so every id from here on would name a "
                "different card than the run pinned. Refusing rather than reinterpreting."
            )

    # -- the log ---------------------------------------------------------------

    def own_play(self, tick: int, card: int, *, accepted: int | None = None) -> None:
        """This seat's play of ``card``, paid and cycled at ``tick``, the tick it RUNS.

        Under a command delay a play is accepted earlier than it runs, and until it runs the
        seat knows its own tap: pass the tick it was accepted as ``accepted``, and between the
        two the fields show it waiting (``own_pending``). Without it the play waits for nothing.
        """
        self._feed(tick, _OWN, card)
        self._wait(accepted, tick, _KIND_PLAY, card, None)

    def enemy_play(self, tick: int, card: int) -> None:
        self._feed(tick, _ENEMY, card)

    def own_press(self, tick: int, elixir: int, *, accepted: int | None = None) -> None:
        """An ability press of this seat's at ``tick``, paid ``elixir`` from its bar; under a
        command delay, ``accepted`` is the tick it was accepted, as for ``own_play``."""
        self._feed_press(tick, _OWN, elixir)
        self._wait(accepted, tick, _KIND_PRESS, -1, int(elixir))

    def _wait(
        self, accepted: int | None, runs: int, kind: int, what: int, cost: int | None
    ) -> None:
        if accepted is None or accepted >= runs:
            return
        if accepted < self.memory.tick:
            raise ValueError(
                f"a command accepted at tick {accepted} arrived after tick {self.memory.tick} "
                "was observed"
            )
        self._waiting.append((int(accepted), int(runs), kind, int(what), cost))

    def enemy_press(self, tick: int, elixir: int) -> None:
        """An ability press of the opponent's at ``tick``, paid ``elixir`` from its bar."""
        self._feed_press(tick, _ENEMY, elixir)

    def _feed_press(self, tick: int, side: int, elixir: int) -> None:
        if tick < self.memory.tick:
            raise ValueError(
                f"a press at tick {tick} arrived after tick {self.memory.tick} was observed"
            )
        if elixir < 0:
            raise ValueError(f"a press costs no less than 0 elixir, got {elixir}")
        self._presses.append((tick, self._fed, side, int(elixir)))
        self._fed += 1

    def _feed(self, tick: int, side: int, card: int) -> None:
        if tick < self.memory.tick:
            raise ValueError(
                f"a play at tick {tick} arrived after tick {self.memory.tick} was observed"
            )
        if not 0 <= card < len(self.cards):
            raise ValueError(f"card id {card} is outside a {len(self.cards)}-card catalogue")
        self._pending.append((tick, self._fed, side, card))
        self._fed += 1

    # -- reading ---------------------------------------------------------------

    @property
    def unaffordable(self) -> tuple[int, int]:
        own, enemy = self.memory.unaffordable
        return own, enemy

    def overtime_at(self, tick: int) -> bool:
        """Whether a match still running at ``tick`` is in overtime."""
        return MatchClock.at(tick, self.calibration).overtime

    def fields(self) -> list[str]:
        """The field names ``observe`` returns, in vector order."""
        return [*FAIR_FIELDS, *(["enemy_last_card"] if self.enemy_last_card else [])]

    def hand_costs(self) -> list[int]:
        """What each own hand slot costs as of the last ``observe``, as the engine prices it.

        A card's listed elixir; for a Mirror, the listed elixir of the card it would copy
        (this side's last play that was not a Mirror) plus one, and -1 while there is none.
        This is ``PlayerState.hand_costs``, rebuilt from the log.
        """
        target = self.mirror_target
        costs = []
        for card in self.hand:
            if card == EMPTY_CARD:
                costs.append(-1)
            elif self.cards[card].placement != Placement.MIRROR:
                costs.append(int(self.cards[card].elixir))
            elif target is None:
                costs.append(-1)
            else:
                costs.append(int(self.cards[target].elixir) + 1)
        return costs

    def _price_at(self, card: int, tick: int) -> int:
        """What a play of ``card`` accepted at ``tick`` holds: its listed elixir, or for a
        Mirror the copy's plus one, the copy being the side's last play that had RUN by then
        (the engine prices a waiting command when it accepts it)."""
        if self.cards[card].placement != Placement.MIRROR:
            return int(self.cards[card].elixir)
        target = None
        for runs, played in self._ran:
            if runs < tick:
                target = played
        return -1 if target is None else int(self.cards[target].elixir) + 1

    def own_pending(self) -> list[list[int]]:
        """This seat's commands accepted and not run as of the last ``observe``, as the engine
        reports them (``PlayerState.pending``): ``[kind, what, x, y, ticks_left, cost]``, kind 0
        a play of card ``what`` and kind 1 a press. The tap's x and y are not in a log and are
        not read by the fields."""
        now = self.memory.tick
        rows = []
        for accepted, runs, kind, what, cost in sorted(self._waiting, key=lambda w: w[1]):
            if accepted < now <= runs:
                price = cost if kind == _KIND_PRESS else self._price_at(what, accepted)
                rows.append([kind, what, 0, 0, runs - now, int(price)])
        return rows

    def observe(self, tick: int) -> dict[str, np.ndarray]:
        """The fields at ``tick``, from every play made before it. The clock only moves on."""
        memory = self.memory
        if tick < memory.tick:
            raise ValueError(f"tick {tick} is before the last observation, {memory.tick}")
        clock = MatchClock.at(tick, self.calibration)
        if tick > memory.tick:
            due = sorted(p for p in self._pending if p[0] < tick)
            self._pending = [p for p in self._pending if p[0] >= tick]
            own: list[tuple[int, int]] = []
            enemy: list[tuple[int, int]] = []
            for when, _, side, card in due:
                if side == _OWN:
                    self._refill_until(when)
                    self._cycle(when, card)
                    own.append((when, card))
                else:
                    enemy.append((when, card))
            self._refill_until(tick)
            pressed = sorted(p for p in self._presses if p[0] < tick)
            self._presses = [p for p in self._presses if p[0] >= tick]
            # Passed only when there are any, so a log without presses still runs on a
            # RoyaleGym whose advance() takes none (before 276c3e9).
            presses = {}
            if pressed:
                presses = {
                    "own_presses": [(w, e) for w, _, side, e in pressed if side == _OWN],
                    "foe_presses": [(w, e) for w, _, side, e in pressed if side == _ENEMY],
                }
            memory.advance(tick, clock.regular_ticks, clock.overtime, own, enemy, **presses)
            memory.show_own_hand(self.hand, self.queue[0])
            self._waiting = [w for w in self._waiting if w[1] >= tick]
        # Passed only when a command waits, so a log without a delay still runs on a RoyaleGym
        # whose fair_fields() takes no own_pending (before 5565645).
        waiting = self.own_pending()
        extra = {"own_pending": waiting} if waiting else {}
        return fair_fields(
            memory,
            clock,
            self.hand,
            self.queue[0],
            self.law.to_milli(memory.own_fine),
            self.cards,
            self.max_mana,
            enemy_last_card=self.enemy_last_card,
            hand_costs=self.hand_costs(),
            **extra,
        )

    # -- internals ---------------------------------------------------------------

    def _cycle(self, tick: int, card: int) -> None:
        """The played card joins the back of the queue and its slot empties; an idle refill
        timer fills the lowest empty slot at once and is set one tick above its period, so the
        refill step of the play's own tick leaves it at the period."""
        if card not in self.hand:
            raise ValueError(
                f"own play of card {card} at tick {tick}, but the hand is {self.hand}: "
                "the log or the deck order is wrong"
            )
        self.queue.append(card)
        self.hand[self.hand.index(card)] = EMPTY_CARD
        if self.refill_ms == 0:
            self._fill()
            self.refill_ms = self._refill_period(tick) + self.law.tick_ms
        if self.cards[card].placement != Placement.MIRROR:
            self.mirror_target = card
            self._ran.append((tick, card))

    def _fill(self) -> None:
        """The front of the queue into the lowest empty hand slot."""
        self.hand[self.hand.index(EMPTY_CARD)] = self.queue.pop(0)

    def _refill_period(self, tick: int) -> int:
        clock = MatchClock.at(tick, self.calibration)
        rate = self.law.rate_at(tick, clock.regular_ticks, clock.overtime)
        return self._refill_periods[rate - 1]

    def _refill_until(self, tick: int) -> None:
        """Run the refill timer's step of every tick before ``tick`` not yet run: it counts
        down, and at 0 it refills the lowest empty slot and restarts at the period."""
        while self._refill_tick < tick:
            if self.refill_ms == 0 and EMPTY_CARD not in self.hand:
                self._refill_tick = tick
                break
            self.refill_ms = max(self.refill_ms - self.law.tick_ms, 0)
            if self.refill_ms == 0 and EMPTY_CARD in self.hand:
                self._fill()
                self.refill_ms = self._refill_period(self._refill_tick)
            self._refill_tick += 1


def _refill_periods(cal: Calibration) -> tuple[int, int, int]:
    """The refill timer's period at each elixir rate, ms; all 0 (an instant refill) when the
    calibration has no refill keys, as on an engine before the timer."""
    try:
        return (
            cal.int("match.HAND_REFILL_MS_1X"),
            cal.int("match.HAND_REFILL_MS_2X"),
            cal.int("match.HAND_REFILL_MS_3X"),
        )
    except KeyError:
        return (0, 0, 0)
