"""``from_replays``: human games replayed in a learner's environment, as shard rows.

The pieces on their own (form suffixes, the deal order, a match's plan), then whole matches in
the engine: each play made at the first decision at or after its tick, in the seat's own frame,
labelled with the slot that held the card; and a match the engine refuses is skipped and counted.
The dataset's own file layout is read from a parquet file written here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
from royaleimitate.replays import base_key, deal_order, from_replays, match_plan, write_matches
from royaleimitate.shards import ShardContext, ShardReader

DECK = ["knight", "archers", "giant", "mini-pekka", "musketeer", "skeletons", "minions-ev1",
        "hog-rider-hero"]
NAMES = ["Knight", "Archer", "Giant", "MiniPekka", "Musketeer", "Skeletons", "Minions", "HogRider"]


def play(side: str, key: str, tick: int, x: int, y: int, index: int) -> dict[str, Any]:
    return {
        "kind": "play_card",
        "side": side,
        "card_key": key,
        "replay_tick_20hz": tick,
        "source_index": index,
        "coordinates": {"native_world_units": {"x": x, "y": y}},
    }


def payload(tag: str, events: list[dict[str, Any]], deck: list[str] = DECK) -> dict[str, Any]:
    side = {"players": [{"deck": [{"card_key": key} for key in deck]}]}
    return {
        "battle": {"team": side, "opponent": side},
        "events": events,
        "source": {"replay_tag": tag},
    }


def test_form_suffixes_come_off_in_any_order() -> None:
    assert base_key("knight-hero") == "knight"
    assert base_key("minions-ev1") == "minions"
    assert base_key("knight-ev1-hero") == "knight"
    assert base_key("the-log") == "the-log"


def test_the_deal_holds_every_play_in_hand() -> None:
    deck = list("abcdefgh")
    # e is played second, so it is the queue's front: a is played and e is drawn.
    order = deal_order(deck, ["a", "e", "b", "f"])
    assert order is not None
    hand, queue = set(order[:4]), order[4:]
    assert "a" in hand and queue[0] == "e"
    assert deal_order(deck, ["a", "b", "c", "d", "e", "f"]) is not None
    # A played card goes behind the four still queued, so it is back after four more plays.
    assert deal_order(deck, ["a", "b", "c", "d", "a"]) is None
    assert deal_order(deck, ["a", "b", "c", "d", "e", "a"]) is not None


def test_a_match_plan_names_why_it_cannot_be_replayed() -> None:
    catalogue = frozenset(NAMES)
    plan = match_plan(payload("ab", [play("team", "knight", 150, 9500, 5500, 0)]), catalogue)
    assert not isinstance(plan, str)
    assert plan.decks[0][:1] == ["Knight"] and sorted(plan.decks[0]) == sorted(NAMES)
    assert plan.plays[0][0].card == "Knight" and plan.plays[1] == []
    assert match_plan(payload("ab", []), catalogue - {"Giant"}) == "card not in the catalogue"
    stranger = [play("team", "zap", 150, 9500, 5500, 0)]
    assert match_plan(payload("ab", stranger), catalogue) == "play of a card not in the deck"
    cycle = [play("team", key, 150 + i, 9500, 5500, i) for i, key in enumerate(
        ["knight", "archers", "giant", "mini-pekka", "knight"])]
    assert match_plan(payload("ab", cycle), catalogue) == "no deal order fits the plays"


needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def build_env():
    from royalegym import ClashParallelEnv
    from royalegym.done_condition import GameOverCondition, StepLimitCondition
    from royalegym.rust_engine import RustEngine

    return ClashParallelEnv(
        RustEngine(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(40),
    )


@pytest.fixture(scope="module")
def learner(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from royalelearn import Learner

    return Learner(build_env, n_envs=1, device="cpu", save_dir=tmp_path_factory.mktemp("run"))


#: Blue plays a Knight at (9, 5) and Archers at (3, 4); Red a Knight at world (8, 26).
MATCH = [
    play("team", "knight", 150, 9500, 5500, 0),
    play("opponent", "knight", 155, 8500, 26500, 1),
    play("team", "archers", 251, 3500, 4500, 2),
]
#: Blue plays a Giant and then a Hog Rider a tick later, which its elixir cannot pay for.
SHORT_OF_ELIXIR = [
    play("team", "giant", 150, 9500, 5500, 0),
    play("team", "hog-rider", 151, 3500, 4500, 1),
]


@needs_engine
def test_each_play_is_made_at_its_decision_and_labelled_with_its_slot(
    learner: Any, tmp_path: Path
) -> None:
    lines: list[str] = []
    directory = write_matches(
        learner,
        tmp_path,
        [payload("a1", MATCH), payload("b2", SHORT_OF_ELIXIR)],
        printer=lines.append,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["producer"]["matches"] == 1
    assert manifest["producer"]["skipped"] == {"play refused by the engine": 1}
    assert "1 play refused by the engine" in lines[-1]

    reader = ShardReader(directory, ShardContext.of_config(learner.config))
    parts = [reader.part(i) for i in range(len(reader.manifest.parts))]
    action, seat, step = (
        np.concatenate([part[k] for part in parts]) for k in ("action", "seat", "tick")
    )
    assert len(action) == 2 * 40  # both seats, every decision until the step limit
    env = build_env()
    parser = env.action_parser
    env.reset(seed=0)
    made = {
        (int(s), int(t)): parser.decode(int(a))
        for a, s, t in zip(action, seat, step, strict=True)
        if a
    }
    # A decision is ten ticks: tick 150 is decision 15, 155 waits for 16, 251 for 26. The
    # first four cards of the deal are slots 0 to 3, and Blue's Knight was played first.
    assert set(made) == {(0, 15), (1, 16), (0, 26)}
    assert made[(0, 15)][1:] == (9, 5)
    assert made[(1, 16)][1:] == (17 - 8, 31 - 26)  # Red's own frame
    assert made[(0, 26)][1:] == (3, 4)


@needs_engine
def test_the_dataset_files_are_read_as_published(learner: Any, tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (tmp_path / "data" / "replays").mkdir(parents=True)
    table = pa.table({"payload_json": [json.dumps(payload("c3", MATCH))]})
    pq.write_table(table, tmp_path / "data" / "replays" / "part-000000.parquet")
    lines: list[str] = []
    directory = from_replays(
        learner, tmp_path / "out", matches=5, data_dir=tmp_path / "data", printer=lines.append
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["producer"]["matches"] == 1 and manifest["producer"]["rows"] == 80
    assert lines[-1].startswith("1 matches written (80 rows)")
