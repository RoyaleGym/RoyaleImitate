"""PublicLogMemory against the env: the same fields from a log as from a battle the env runs.

The memory's promise is that a model trained on fields rebuilt from a log reads the
numbers the env would have shown it. So the check is the env itself: play a battle,
log every accepted play as (tick, team, card), and at every step compare each field
the log memory returns with the same slice of the vector the env's builder writes, for
both seats, with and without ``enemy_last_card``. The field names and their order must
be the env's as well. A mismatch names tick, seat and field.

The battles cover what the assumptions in public_log.py depend on: a full turn of both
queues, the 1x-to-2x switch, the tick regulation ends and the overtime after it,
observation gaps of 1 to 13 ticks, a seat that holds at a full bar (so leak and the
pay-before-regen order both show), and a seat that plays often. Each coverage claim is
asserted, not assumed. The memory-only tests add two own plays inside one gap and a
calibration that is not the default.

The last tests hold the catalogue pin: a card list that differs from the pinned one by
a card inserted, appended, missing or renamed, or by two cards swapped, is refused by
name and id.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import (
    BLUE,
    DECK_SIZE,
    HAND_SIZE,
    TEAMS,
    DeployStatus,
    ElixirLaw,
    MatchSetup,
    ShuffleMode,
    default_calibration,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royaleimitate.public_log import BOARD_FIELDS, PublicLogMemory

DECKS = (
    ["Skeletons", "Knight", "Archer", "Goblins", "Minions", "Fireball", "Musketeer", "Valkyrie"],
    ["Goblins", "Valkyrie", "Knight", "Fireball", "Skeletons", "Musketeer", "Archer", "Minions"],
)
CADENCES = (10, 7, 1, 13, 4)
#: Chance a seat plays on a step it may play. Blue may play only from a full bar, so it
#: leaks while it waits; Red may play whenever the mask allows.
PLAY = {0: 0.25, 1: 0.8}
FULL_ONLY = {0: True, 1: False}

ENGINES = {
    "mock": MockEngine,
    "rust": pytest.param(
        RustEngine, marks=pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
    ),
}


class Played(NamedTuple):
    mismatches: list[str]
    plays: dict[int, int]
    unaffordable: dict[int, tuple[int, int]]
    seen: dict[str, Any]


def ids(engine) -> dict[str, int]:
    return {c.name: i for i, c in enumerate(engine.cards())}


def play_out(engine, seed: int, start_tick: int, ticks: int, shift: int = 0) -> Played:
    """One battle, the env's fields and the log's side by side at every step.

    ``shift`` moves Blue's own plays that many ticks later in Blue's log, and nothing
    else: the plant.
    """
    card = ids(engine)
    decks = [[card[n] for n in d] for d in DECKS]
    max_mana = default_calibration().int("match.MAX_MANA")
    rng = np.random.default_rng(seed)
    parser = TileActionParser()
    parser.bind(engine)
    builders = {False: SpatialObsBuilder(), True: SpatialObsBuilder(card_identity=True)}
    for b in builders.values():
        b.bind(engine, parser)
    engine.reset(seed, MatchSetup(decks=decks, shuffle=ShuffleMode.NONE, start_tick=start_tick))
    state = engine.state()
    for b in builders.values():
        b.reset(state)
    # The pin is read back from the engine the battle runs on. A run must not do that,
    # since the check would then compare the catalogue with itself. It is fine here:
    # these battles test the fields, and the pin has its own tests at the end.
    names = [c.name for c in engine.cards()]
    logs = {
        (team, flag): PublicLogMemory(
            engine.cards(),
            decks[team],
            card_names=names,
            start_tick=state.tick,
            own_elixir_milli=state.players[team].elixir_milli,
            enemy_elixir_milli=state.players[1 - team].elixir_milli,
            enemy_last_card=flag,
        )
        for team in TEAMS
        for flag in builders
    }
    seen: dict[str, Any] = {
        "dealt": [(list(state.players[t].hand), state.players[t].next_card) for t in TEAMS],
        "decks": decks,
        "rates": set(),
        "overtime": set(),
        "leaked": False,
        "narrowed": False,
        "ticks": set(),
    }
    mismatches: list[str] = []
    plays = {t: 0 for t in TEAMS}
    step = 0
    end = state.tick + ticks
    while not state.game_over and state.tick < end:
        seen["rates"].add(state.elixir_rate)
        seen["overtime"].add(state.overtime)
        seen["ticks"].add(state.tick)
        for (team, flag), log in logs.items():
            vec = builders[flag].build(state, team, parser.action_mask(state, team))["vector"]
            offsets = builders[flag].vector_offsets()
            got = log.observe(state.tick)
            where = f"tick {state.tick} seat {team} card_identity={flag}"
            # The env's fields but the board's, in the env's order: a field left out, or
            # the right fields in another order, would scramble a vector built from them.
            keys = [k for k in offsets if k not in BOARD_FIELDS]
            if list(got) != keys or log.fields() != keys:
                mismatches.append(
                    f"{where} names: observe {list(got)} fields() {log.fields()} env {keys}"
                )
            if log.overtime_at(state.tick) != state.overtime:
                mismatches.append(
                    f"{where} overtime_at {log.overtime_at(state.tick)} env {state.overtime}"
                )
            for field, value in got.items():
                want = vec[offsets[field]]
                if not np.array_equal(value, want):
                    mismatches.append(f"{where} {field}: log {value.tolist()} env {want.tolist()}")
            seen["leaked"] |= bool(got["own_elixir_leaked"][0] > 0)
            seen["narrowed"] |= bool(0 < got["enemy_possible_hand"].sum() < DECK_SIZE)
        commands = []
        for team in TEAMS:
            full = state.players[team].elixir_milli >= 1000 * max_mana
            if (FULL_ONLY[team] and not full) or rng.random() >= PLAY[team]:
                continue
            legal = np.flatnonzero(parser.action_mask(state, team)[1:]) + 1
            if len(legal):
                cmd = parser.parse(int(rng.choice(legal)), state, team)
                if cmd is not None:
                    commands.append(cmd)
        for r in engine.step(commands, CADENCES[step % len(CADENCES)]):
            if r.status != DeployStatus.OK:
                continue
            plays[r.team] += 1
            for (team, _flag), log in logs.items():
                if r.team == team:
                    log.own_play(r.tick + (shift if team == BLUE else 0), r.card_id)
                else:
                    log.enemy_play(r.tick, r.card_id)
        step += 1
        state = engine.state()
    return Played(mismatches, plays, {k[0]: v.unaffordable for k, v in logs.items()}, seen)


@pytest.fixture(scope="module", params=list(ENGINES.values()), ids=list(ENGINES))
def engine(request):
    return request.param()


def regular_ticks() -> int:
    cal = default_calibration()
    return -(-cal.int("match.REGULAR_TIME_S") * 1000 // cal.int("time.TICK_MS"))


def test_the_log_gives_the_env_fields_through_a_whole_opening(engine):
    run = play_out(engine, seed=11, start_tick=engine.rules().deploy_lockout_ticks, ticks=2400)
    assert not run.mismatches, "\n".join(run.mismatches[:10])
    assert run.plays[1] > DECK_SIZE, f"Red played {run.plays[1]}: its queue never came round"
    assert run.plays[0] >= HAND_SIZE, f"Blue played {run.plays[0]}"
    assert run.seen["leaked"], "no seat ever leaked, so the leak field compared only zeros"
    assert run.seen["narrowed"], "the enemy's possible hand never narrowed below the catalogue"
    assert run.unaffordable == {0: (0, 0), 1: (0, 0)}, run.unaffordable


def test_the_log_gives_the_env_fields_across_the_switch_to_2x(engine):
    start = regular_ticks() - ElixirLaw.load().speedup_ticks - 150  # 1x for 150 ticks
    run = play_out(engine, seed=5, start_tick=start, ticks=600)
    assert not run.mismatches, "\n".join(run.mismatches[:10])
    assert run.seen["rates"] == {1, 2}, f"rates seen {run.seen['rates']}"
    assert run.unaffordable == {0: (0, 0), 1: (0, 0)}, run.unaffordable


def test_the_log_gives_the_env_fields_across_the_end_of_regulation(engine):
    """Started one turn of the cadences before the end, from an empty board: nobody can
    take a crown that fast, so regulation ends level and the battle goes on into overtime.
    One turn of the cadences is what puts an observation on the tick regulation ends."""
    run = play_out(engine, seed=5, start_tick=regular_ticks() - sum(CADENCES), ticks=400)
    assert not run.mismatches, "\n".join(run.mismatches[:10])
    assert regular_ticks() in run.seen["ticks"], "the tick regulation ends was never observed"
    assert run.seen["overtime"] == {False, True}, (
        f"overtime seen {run.seen['overtime']}: the battle never crossed the end of regulation"
    )
    assert run.plays[1] > 0, "nobody played, so the overtime clock was compared on its own"
    assert run.unaffordable == {0: (0, 0), 1: (0, 0)}, run.unaffordable


def test_a_deck_dealt_without_a_shuffle_is_hand_then_queue(engine):
    """The deck order the memory is given means what the engine means by it."""
    run = play_out(engine, seed=1, start_tick=engine.rules().deploy_lockout_ticks, ticks=1)
    for team in TEAMS:
        deck = run.seen["decks"][team]
        assert run.seen["dealt"][team] == (deck[:HAND_SIZE], deck[HAND_SIZE])


def test_plant_a_log_one_tick_late_is_caught(engine):
    """Blue's plays logged one tick after they happened: the bar pays after a tick of
    regeneration instead of before it, which a full bar turns into a different leak."""
    run = play_out(
        engine, seed=11, start_tick=engine.rules().deploy_lockout_ticks, ticks=2400, shift=1
    )
    assert run.mismatches, "PLANT DID NOT LAND: a one-tick-late log matched the env"
    assert all(" seat 0 " in m for m in run.mismatches), run.mismatches[:5]


# -- the memory alone ---------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_cards():
    return MockEngine().cards()


def deck_of(cards, names) -> list[int]:
    by = {c.name: i for i, c in enumerate(cards)}
    return [by[n] for n in names]


def names_of(cards) -> list[str]:
    """The pin, read back from the catalogue. Fine for these tests, which are about the
    memory; a run passes the list its config names, as the pin tests below do."""
    return [c.name for c in cards]


def test_splitting_the_clock_changes_nothing_but_the_seen_tick(mock_cards):
    """Observed every tick or every 10, with plays in between: the same fields.

    A play often comes the moment a card is affordable, which is rarely on an observation
    tick. Both bars start empty, so at 1x (one elixir per 56 ticks) a Knight becomes
    affordable at tick 168 exactly, and the Knight is played then: the last observation
    before it, at 160, sees 2.86 elixir. Paying at 160 would floor the bar at zero and
    hand back 0.14 elixir that was never there, for the rest of the match. The Archer
    at 745 comes 17 ticks after the bar filled, so it checks the leak as well.

    Only ``own_ticks_since_play`` may differ, because it counts from the observation
    that first showed the play, which is what the env's memory has always done.
    """
    deck = deck_of(mock_cards, DECKS[0])
    fine, coarse = (
        PublicLogMemory(
            mock_cards,
            deck,
            card_names=names_of(mock_cards),
            own_elixir_milli=0,
            enemy_elixir_milli=0,
        )
        for _ in range(2)
    )
    assert [mock_cards[c].name for c in deck[1:3]] == ["Knight", "Archer"]
    for log in (fine, coarse):
        log.own_play(168, deck[1])
        log.own_play(745, deck[2])
        log.enemy_play(168, deck[1])
    for tick in range(1, 901):
        a = fine.observe(tick)
        if tick % 10 == 0:
            b = coarse.observe(tick)
            for field in a:
                if field != "own_ticks_since_play":
                    assert np.array_equal(a[field], b[field]), (tick, field, a[field], b[field])
    assert fine.unaffordable == coarse.unaffordable == (0, 0)
    assert coarse.observe(900)["own_elixir_leaked"][0] > 0, "the bar never filled"


def test_a_play_at_the_observed_tick_is_not_seen_yet(mock_cards):
    """An observation at T is taken before tick T runs; a play at T is paid as it runs."""
    deck = deck_of(mock_cards, DECKS[0])
    log = PublicLogMemory(mock_cards, deck, card_names=names_of(mock_cards))
    log.own_play(100, deck[0])
    log.enemy_play(100, deck[3])
    at = log.observe(100)
    after = log.observe(101)
    assert at["enemy_plays"][0] == 0, "an enemy play was seen at the tick it was made"
    assert after["enemy_plays"][0] > 0
    assert at["own_last_card"][-1] == 1, "an own play was seen at the tick it was made"
    assert after["own_last_card"][deck[0]] == 1


def test_a_play_the_bar_cannot_pay_is_counted(mock_cards):
    deck = deck_of(mock_cards, DECKS[0])
    log = PublicLogMemory(mock_cards, deck, card_names=names_of(mock_cards), own_elixir_milli=1000)
    log.own_play(0, deck[1])  # a 3-elixir Knight on 1 elixir
    log.observe(1)
    assert log.unaffordable == (1, 0)


def test_the_board_fields_are_left_out_and_the_rest_are_there(mock_cards):
    log = PublicLogMemory(
        mock_cards,
        deck_of(mock_cards, DECKS[0]),
        card_names=names_of(mock_cards),
        enemy_last_card=True,
    )
    assert not set(BOARD_FIELDS) & set(log.observe(0))
    assert log.fields()[-1] == "enemy_last_card"
    assert {"own_elixir", "clock"} <= set(log.fields())


@pytest.mark.parametrize(
    ("deck", "message"),
    [
        (lambda d: d[:7], "8 distinct"),
        (lambda d: [d[0], *d[:7]], "8 distinct"),
        (lambda d: [*d[:7], 999], "outside"),
    ],
    ids=["seven", "duplicate", "unknown"],
)
def test_a_deck_that_is_not_eight_distinct_known_cards_is_refused(mock_cards, deck, message):
    with pytest.raises(ValueError, match=message):
        PublicLogMemory(
            mock_cards, deck(deck_of(mock_cards, DECKS[0])), card_names=names_of(mock_cards)
        )


def test_a_log_that_contradicts_the_deck_is_refused(mock_cards):
    deck = deck_of(mock_cards, DECKS[0])
    log = PublicLogMemory(mock_cards, deck, card_names=names_of(mock_cards))
    log.own_play(100, deck[5])  # the second card in the queue, not in hand
    with pytest.raises(ValueError, match="the hand is"):
        log.observe(101)


def test_the_clock_only_moves_on(mock_cards):
    deck = deck_of(mock_cards, DECKS[0])
    log = PublicLogMemory(mock_cards, deck, card_names=names_of(mock_cards))
    first = log.observe(50)
    again = log.observe(50)
    assert all(np.array_equal(first[k], again[k]) for k in first)
    with pytest.raises(ValueError, match="before the last observation"):
        log.observe(49)
    with pytest.raises(ValueError, match="arrived after tick 50"):
        log.own_play(40, deck[0])


def test_two_own_plays_in_one_gap_cycle_in_the_order_made(mock_cards):
    """Two own plays between two observations, fed newest first.

    Observed every tick, each gap holds one play, so the order they were fed cannot
    matter. Observed every 50 ticks, one gap holds both, and the hand is right only if
    they cycle in the order they were made.
    """
    deck = deck_of(mock_cards, DECKS[0])
    fine, coarse = (
        PublicLogMemory(mock_cards, deck, card_names=names_of(mock_cards)) for _ in range(2)
    )
    for log in (fine, coarse):
        log.own_play(120, deck[1])  # fed first, made second
        log.own_play(110, deck[0])
    for tick in range(1, 151):
        a = fine.observe(tick)
        if tick % 50 == 0:
            b = coarse.observe(tick)
            for field in a:
                if field != "own_ticks_since_play":
                    assert np.array_equal(a[field], b[field]), (tick, field, a[field], b[field])
    assert fine.hand == coarse.hand == [deck[4], deck[5], deck[2], deck[3]]
    assert fine.unaffordable == coarse.unaffordable == (0, 0)


def test_the_clock_reads_the_calibration_given():
    """Regulation halved: the log turns to overtime on the tick an engine loaded with the
    same calibration does, not on the default's."""
    base = default_calibration()
    cal = base.with_override("match.REGULAR_TIME_S", base.int("match.REGULAR_TIME_S") // 2)
    engine = MockEngine(calibration=cal)
    cards = engine.cards()
    decks = [deck_of(cards, d) for d in DECKS]
    engine.reset(0, MatchSetup(decks=decks, shuffle=ShuffleMode.NONE, start_tick=0))
    end = engine.state().regular_ticks
    assert end < regular_ticks(), "the calibration did not shorten regulation"
    engine.reset(0, MatchSetup(decks=decks, shuffle=ShuffleMode.NONE, start_tick=end - 1))
    state = engine.state()
    log = PublicLogMemory(
        cards,
        decks[0],
        card_names=names_of(cards),
        calibration=cal,
        start_tick=state.tick,
        own_elixir_milli=state.players[0].elixir_milli,
        enemy_elixir_milli=state.players[1].elixir_milli,
    )
    seen = []
    for _ in range(2):
        got = log.observe(state.tick)
        seen.append(state.overtime)
        assert bool(got["clock"][1]) == state.overtime, (state.tick, got["clock"])
        assert log.overtime_at(state.tick) == state.overtime, state.tick
        engine.step([], 1)
        state = engine.state()
    assert seen == [False, True], seen


def test_the_class_is_this_package_s_own():
    """The class and every base it has are defined here, not borrowed from RoyaleGym."""
    assert PublicLogMemory.__module__ == "royaleimitate.public_log"
    borrowed = [c for c in PublicLogMemory.__mro__ if c.__module__.startswith("royalegym")]
    assert not borrowed, borrowed


# -- the catalogue pin ----------------------------------------------------------------

#: The catalogue a run pins, as its config would name it. Written out here and never read
#: from an engine, so each check below compares two lists that were made apart.
PINNED = (
    "Knight",
    "Archer",
    "Goblins",
    "Giant",
    "Musketeer",
    "Skeletons",
    "Minions",
    "Valkyrie",
    "Fireball",
    "Arrows",
)
#: A deck of ids 0 to 7: valid in every catalogue ``refusal`` is given, so a refusal there
#: can only be the pin's.
PIN_DECK = list(range(DECK_SIZE))


def loaded(names) -> list:
    """A MockEngine catalogue of exactly ``names``, in that order."""
    cards = MockEngine(card_names=names).cards()
    assert [c.name for c in cards] == list(names), "the engine did not load the list asked"
    return cards


def refusal(names) -> str:
    """The message a memory pinned to PINNED gives for a catalogue of ``names``.

    The same cards pinned to their own names are accepted first, so the refusal that
    follows can only come from the pin, not from the deck or the cards.
    """
    cards = loaded(names)
    PublicLogMemory(cards, PIN_DECK, card_names=list(names))
    with pytest.raises(ValueError) as caught:
        PublicLogMemory(cards, PIN_DECK, card_names=PINNED)
    return str(caught.value)


def test_the_pin_is_required():
    with pytest.raises(TypeError, match=r"required keyword-only argument: 'card_names'"):
        PublicLogMemory(loaded(PINNED), PIN_DECK)  # type: ignore[call-arg]


def test_the_exact_pinned_catalogue_is_accepted():
    """PINNED is a tuple and the names read from the cards are a list: a pin compared
    without making both lists would refuse its own catalogue here."""
    log = PublicLogMemory(loaded(PINNED), PIN_DECK, card_names=PINNED)
    assert log.card_names == list(PINNED)
    assert [c.name for c in log.cards] == list(PINNED)
    assert set(log.observe(1)) == set(log.fields())


def test_a_card_inserted_in_the_middle_is_refused_at_the_first_shifted_id():
    """The real hazard: one more loadable card renumbers every id after it."""
    message = refusal([*PINNED[:4], "HogRider", *PINNED[4:]])
    assert "has 'Musketeer' at id 4" in message, message
    assert "have 'HogRider' there" in message, message
    assert "(10 pinned names, 11 loaded)" in message, message
    assert "Catalogue ids are positional" in message, message


def test_a_card_appended_at_the_end_is_refused():
    message = refusal([*PINNED, "HogRider"])
    assert "has '<end>' at id 10" in message, message
    assert "have 'HogRider' there" in message, message
    assert "(10 pinned names, 11 loaded)" in message, message


def test_two_cards_swapped_are_refused():
    """Same length, same set of names: only the order differs."""
    message = refusal([PINNED[0], PINNED[2], PINNED[1], *PINNED[3:]])
    assert "has 'Archer' at id 1" in message, message
    assert "have 'Goblins' there" in message, message
    assert "(10 pinned names, 10 loaded)" in message, message


def test_a_renamed_card_is_refused():
    """Same length, one name different, at the last id: not a reorder and not a length change."""
    message = refusal([*PINNED[:9], "HogRider"])
    assert "has 'Arrows' at id 9" in message, message
    assert "have 'HogRider' there" in message, message
    assert "(10 pinned names, 10 loaded)" in message, message


def test_a_missing_card_is_reported_as_the_catalogue_not_the_deck():
    """The pin is checked before any id is read. Here the deck names id 9, past the end
    of a nine-card catalogue, and the refusal must still be the catalogue's."""
    with pytest.raises(ValueError) as caught:
        PublicLogMemory(loaded(PINNED[:9]), list(range(2, 10)), card_names=PINNED)
    message = str(caught.value)
    assert "has 'Arrows' at id 9" in message, message
    assert "have '<end>' there" in message, message
    assert "(10 pinned names, 9 loaded)" in message, message
