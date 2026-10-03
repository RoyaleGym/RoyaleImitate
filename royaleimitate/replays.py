"""Human games as demonstrations: the IL_Replay dataset replayed in a learner's environment.

``from_replays`` downloads IL_Replay (``VanguardX101/IL_Replay`` on Hugging Face: ladder games,
anonymised) into the Hugging Face cache, replays each match in a
``royalelearn.Learner``'s own environment with both players' card plays, and stores every
decision of both seats as a shard row (``royaleimitate.shards``), labelled with what the human
did there: the card and tile they played, or the no-op. ``clone`` reads the directory it returns.

How a match is replayed:

- Decks. A deck entry's card key loses its form suffix (``-ev1``, ``-hero``) and is mapped to
  the engine's card name (``replay_cards.json``). A match with a card the environment's
  catalogue does not hold is skipped. Every card plays in its base form.
- The deal. The dataset does not say which four cards a player started with, so each deck is
  dealt unshuffled in an order under which every play of the match was in hand: the first four
  cards are the hand, and a played card goes to the back of the queue. A match that no order
  explains is skipped.
- Time. A play recorded at tick ``t`` is made at the first decision at or after ``t``. A seat
  makes one play per decision, so a second play in the same decision waits for the next one.
  Every other decision is labelled the no-op, until the engine's battle ends.
- Ability presses are not replayed.
- A match where the engine refuses one of the plays (for want of elixir, say) is skipped.
  Under a command delay with the action parser's ``hold_while_pending``, a seat with a play
  waiting is offered only the no-op, so a second play made before the first lands is refused
  and its match skipped. ``make_env`` has neither by default.

The skips are counted, printed, and kept in the shard manifest's ``producer`` entry.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from royalelearn.extensions import PreflightError

from .shards import ShardContext, ShardWriter

__all__ = ["DATASET", "deal_order", "from_replays", "match_plan", "write_matches"]

#: The dataset on Hugging Face.
DATASET = "VanguardX101/IL_Replay"
#: Dataset card key -> engine card name. Card keys are matched, never display names.
CARD_NAMES: dict[str, str] = json.loads(
    (Path(__file__).with_name("replay_cards.json")).read_text(encoding="utf-8")
)
_FORM_SUFFIXES = ("-ev1", "-hero")
#: The dataset's world units per tile, on both axes.
_UNITS_PER_TILE = 1000
_SIDES = {"team": 0, "opponent": 1}
_AGENTS = ("blue", "red")
_INSTALL = 'pip install "royaleimitate[replays]"'


class Play(NamedTuple):
    tick: int
    card: str
    x: int
    y: int


class MatchPlan(NamedTuple):
    """One match as the engine replays it: two dealt decks and each seat's plays in order."""

    tag: str
    decks: tuple[list[str], list[str]]
    plays: tuple[list[Play], list[Play]]


def base_key(key: str) -> str:
    """A deck entry's card key without its form suffix: ``knight-hero`` -> ``knight``."""
    changed = True
    while changed:
        changed = False
        for suffix in _FORM_SUFFIXES:
            if key.endswith(suffix):
                key, changed = key[: -len(suffix)], True
    return key


def deal_order(deck: Sequence[str], plays: Sequence[str]) -> list[str] | None:
    """An order of ``deck`` under which every card of ``plays`` is in hand when played, or None.

    The first four cards are the hand and the rest the queue, in order; a played card goes to
    the back of the queue and the queue's front card joins the hand.
    """
    deck = list(deck)
    for hand in combinations(deck, 4):
        rest = [card for card in deck if card not in hand]
        for queue in permutations(rest):
            held, waiting = set(hand), list(queue)
            for card in plays:
                if card not in held:
                    break
                held.remove(card)
                waiting.append(card)
                held.add(waiting.pop(0))
            else:
                return [*hand, *queue]
    return None


def match_plan(payload: dict[str, Any], catalogue: frozenset[str]) -> MatchPlan | str:
    """The plan for one match's ``payload_json``, or why it cannot be replayed."""
    battle = payload["battle"]
    decks: list[list[str]] = []
    for side in _SIDES:
        players = battle[side]["players"]
        if len(players) != 1:
            return "not one player a side"
        names = [CARD_NAMES.get(base_key(card["card_key"])) for card in players[0]["deck"]]
        if len(names) != 8 or len(set(names)) != 8:
            return "not an eight-card deck"
        if any(name not in catalogue for name in names):
            return "card not in the catalogue"
        decks.append([str(name) for name in names])
    plays: list[list[Play]] = [[], []]
    for event in sorted(
        payload["events"], key=lambda e: (e["replay_tick_20hz"], e["source_index"])
    ):
        if event["kind"] != "play_card":
            continue
        seat = _SIDES[event["side"]]
        name = CARD_NAMES.get(base_key(event["card_key"]))
        if name not in decks[seat]:
            return "play of a card not in the deck"
        units = event["coordinates"]["native_world_units"]
        plays[seat].append(Play(int(event["replay_tick_20hz"]), name, units["x"], units["y"]))
    dealt = []
    for seat in (0, 1):
        order = deal_order(decks[seat], [play.card for play in plays[seat]])
        if order is None:
            return "no deal order fits the plays"
        dealt.append(order)
    return MatchPlan(payload["source"]["replay_tag"], (dealt[0], dealt[1]), (plays[0], plays[1]))


def _payloads(
    data_dir: str | Path | None, cache_dir: str | Path | None, revision: str | None
) -> tuple[Iterator[dict[str, Any]], str]:
    """Every match's payload, part by part, downloading each part when it is first needed."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PreflightError(f"reading IL_Replay needs pyarrow: {_INSTALL}") from error
    if data_dir is not None:
        parts: list[Path] = sorted(Path(data_dir).glob("replays/*.parquet"))
        if not parts:
            raise PreflightError(f"{data_dir} has no replays/*.parquet")
        source = str(data_dir)

        def fetch(part: Any) -> Path:
            return Path(part)

    else:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as error:
            raise PreflightError(
                f"downloading IL_Replay needs huggingface_hub: {_INSTALL}"
            ) from error
        info = HfApi().dataset_info(DATASET, revision=revision)
        revision = info.sha
        parts = sorted(
            s.rfilename
            for s in info.siblings or []
            if s.rfilename.startswith("replays/") and s.rfilename.endswith(".parquet")
        )
        source = f"{DATASET}@{revision}"

        def fetch(part: Any) -> Path:
            return Path(
                hf_hub_download(
                    DATASET, part, repo_type="dataset", revision=revision, cache_dir=cache_dir
                )
            )

    def payloads() -> Iterator[dict[str, Any]]:
        for part in parts:
            table = pq.ParquetFile(fetch(part))
            for batch in table.iter_batches(columns=["payload_json"], batch_size=256):
                for text in batch.column(0).to_pylist():
                    yield json.loads(text)

    return payloads(), source


def _battle_env(env: Any) -> Any:
    """The ``royalegym`` parallel environment inside ``env`` (the one with ``battle_state``)."""
    from royalegym.env import ClashParallelEnv

    seen = env
    for _ in range(8):
        if isinstance(seen, ClashParallelEnv):
            return seen
        seen = getattr(seen, "parallel", None) or getattr(seen, "env", None)
        if seen is None:
            break
    raise PreflightError("the learner's environment is not a royalegym battle environment")


def _replay(env: Any, battle: Any, plan: MatchPlan, seed: int) -> list[tuple] | str:
    """Every decision of both seats as (agent, obs, action, step), or why the engine refused."""
    from royalegym.protocol import EMPTY_CARD, MatchSetup, ShuffleMode

    names = {card.name: card.card_id for card in battle.engine.cards()}
    parser = battle.action_parser
    tiles_x, tiles_y = parser.nx // parser.pitch_div, parser.ny // parser.pitch_div
    setup = MatchSetup(
        decks=[[names[name] for name in deck] for deck in plan.decks],
        shuffle=int(ShuffleMode.NONE),
    )
    obs, _info = env.reset(seed=seed, options={"setup": setup})
    waiting = [deque(plan.plays[0]), deque(plan.plays[1])]
    rows: list[tuple] = []
    step = 0
    while env.agents:
        state = battle.battle_state
        actions = {}
        for seat, agent in enumerate(_AGENTS):
            action = 0
            queue = waiting[seat]
            if queue and queue[0].tick <= state.tick:
                play = queue.popleft()
                hand = list(state.players[seat].hand)
                card = names[play.card]
                if card not in hand or card == EMPTY_CARD:
                    return "card not in hand"
                x = min(play.x * parser.nx // (tiles_x * _UNITS_PER_TILE), parser.nx - 1)
                y = min(play.y * parser.ny // (tiles_y * _UNITS_PER_TILE), parser.ny - 1)
                if seat == 1:  # Red acts in its own frame, turned half a turn
                    x, y = parser.nx - 1 - x, parser.ny - 1 - y
                action = parser.encode(hand.index(card), x, y)
                if not obs[agent]["action_mask"][action]:
                    return "play refused by the engine"
            actions[agent] = action
            rows.append((agent, {k: np.array(v) for k, v in obs[agent].items()}, action, step))
        obs, _rewards, _terminated, _truncated, _info = env.step(actions)
        step += 1
    return rows


def from_replays(
    learner: Any,
    out: str | Path,
    *,
    matches: int = 1000,
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    printer: Callable[[str], None] = print,
) -> Path:
    """Replay ``matches`` human games of IL_Replay in ``learner``'s environment and store both
    seats' decisions as shard rows. Returns the shard directory (under ``out``) to give ``clone``.

    The dataset's parts download into the Hugging Face cache (``cache_dir`` to choose another
    folder) as they are needed: one part holds about 5,000 matches. ``data_dir`` reads a copy
    you already have instead (a folder holding ``replays/*.parquet``). ``matches`` counts the
    matches written; skipped ones do not count, and the dataset is read in its own order.
    """
    payloads, source = _payloads(data_dir, cache_dir, revision)
    return write_matches(learner, out, payloads, matches=matches, source=source, printer=printer)


def write_matches(
    learner: Any,
    out: str | Path,
    payloads: Iterable[dict[str, Any]],
    *,
    matches: int = 1000,
    source: str = "",
    printer: Callable[[str], None] = print,
) -> Path:
    """``from_replays`` on matches you supply: each one an IL_Replay ``payload_json``, parsed."""
    env = learner.build_env()
    battle = _battle_env(env)
    catalogue = frozenset(card.name for card in battle.engine.cards())
    context = ShardContext.of_config(learner.config)
    skipped: Counter[str] = Counter()
    written = rows = 0
    producer = {"tool": "royaleimitate.from_replays", "dataset": source}
    try:
        with ShardWriter(out, context, producer=producer) as writer:
            for index, payload in enumerate(payloads):
                if written >= matches:
                    break
                plan = match_plan(payload, catalogue)
                if isinstance(plan, str):
                    skipped[plan] += 1
                    continue
                replayed = _replay(env, battle, plan, seed=index)
                if isinstance(replayed, str):
                    skipped[replayed] += 1
                    continue
                group = int(plan.tag[:15], 16)
                for agent, obs, action, step in replayed:
                    writer.add(
                        obs,
                        action=action,
                        weight=1.0,
                        group=group,
                        seat=_AGENTS.index(agent),
                        tick=step,
                    )
                written += 1
                rows += len(replayed)
                if written % 100 == 0:
                    printer(f"{written} matches written, {sum(skipped.values())} skipped")
            writer.manifest.producer.update(
                {"matches": written, "rows": rows, "skipped": dict(sorted(skipped.items()))}
            )
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    reasons = ", ".join(f"{count} {why}" for why, count in skipped.most_common()) or "none"
    printer(f"{written} matches written ({rows} rows); skipped: {reasons}")
    return Path(out) / context.engine_key
