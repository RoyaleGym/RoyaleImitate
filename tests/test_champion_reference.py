"""A ``champion`` reference: the ladder's champion, followed from one promotion to the next.

The reference-KL term reads it like a snapshot reference. At the start of each update it asks
the run for the ladder's champion and, when that is not the one it holds, loads the pool's copy
of its weights into its own actor; until the ladder has a champion it is ``start``, a snapshot
reference of the section. The coefficient's budget controller is the term's, unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royaleimitate.config import ImitationSection, imitation_problems
from royaleimitate.extension import verify_references
from royaleimitate.references import ChampionReference, SnapshotReference, build_references
from royaleimitate.regularisers import ReferenceKL
from royalelearn.errors import PreflightError
from royalelearn.testing import build_model, coordinator, tiny_config

torch = pytest.importorskip("torch")

from imitation_support import seeded_artifact, with_imitation  # noqa: E402


def _section(references: dict[str, Any], reference: str = "champ") -> ImitationSection:
    return msgspec.convert(
        {
            "references": references,
            "regularisers": [
                {"kind": "reference_kl", "name": "champ", "reference": reference}
            ],
        },
        type=ImitationSection,
    )


SNAPSHOT = {"kind": "snapshot", "path": "x", "sha256": "0" * 64}


@pytest.mark.parametrize(
    "references,words",
    [
        ({"champ": {"kind": "champion", "start": "bc"}}, "start 'bc' must name a snapshot"),
        (
            {
                "champ": {"kind": "champion", "start": "bc"},
                "bc": {"kind": "field_mlp", "path": "x", "sha256": "0" * 64},
            },
            "start 'bc' must name a snapshot",
        ),
        (
            {
                "champ": {"kind": "champion", "start": "other"},
                "other": {"kind": "champion", "start": "bc"},
                "bc": SNAPSHOT,
            },
            "start 'other' must name a snapshot",
        ),
    ],
)
def test_a_start_that_is_not_a_snapshot_reference_is_named(
    references: dict[str, Any], words: str
) -> None:
    assert any(words in problem for problem in imitation_problems(_section(references)))


def test_a_champion_needs_no_folder() -> None:
    section = _section({"champ": {"kind": "champion", "start": "bc"}, "bc": SNAPSHOT})
    assert not any("champ" in problem for problem in imitation_problems(section))
    assert not any("champ" in problem for problem in verify_references(section))


class _Ladder:
    """A champion chain that grows when told, and a store that counts its loads."""

    def __init__(self, weights: dict[str, Any]) -> None:
        self.chain: list[str] = []
        self.weights = weights
        self.loads: list[str] = []

    def champion(self, gated: bool = False) -> str | None:
        if not self.chain or (gated and len(self.chain) == 1):
            return None
        return self.chain[-1]

    def load(self, snapshot_id: str, actor: Any) -> None:
        self.loads.append(snapshot_id)
        actor.load_state_dict(self.weights[snapshot_id])


@pytest.mark.parametrize("follow_first", [False, True])
def test_the_reference_is_start_until_a_champion_then_follows_it(
    env_spec: Any, follow_first: bool
) -> None:
    """``start`` until there is a champion to follow, then each champion once. The first,
    promoted without a game, is followed only under ``follow_first``."""
    from test_policy_transform import _obs

    start = SnapshotReference("bc", build_model(env_spec, seed=1).actor)
    zero, one, two = (build_model(env_spec, seed=seed).actor for seed in (2, 3, 5))
    ladder = _Ladder(
        {
            "snap:v0": zero.state_dict(),
            "snap:v1": one.state_dict(),
            "snap:v2": two.state_dict(),
        }
    )
    reference = ChampionReference(
        "champ",
        start,
        build_model(env_spec, seed=4).actor,
        champion=ladder.champion,
        load_snapshot=ladder.load,
        printer=None,
        follow_first=follow_first,
    )
    obs = _obs(env_spec, rows=6)
    assert reference.follow() is None
    assert torch.equal(reference.log_probs(obs), start.log_probs(obs))
    ladder.chain.append("snap:v0")
    if follow_first:
        assert reference.follow() == "snap:v0"
        assert torch.equal(reference.log_probs(obs), zero.distribution(obs).log_probs)
    else:
        assert reference.follow() is None
        assert torch.equal(reference.log_probs(obs), start.log_probs(obs))
    for champion, actor in (("snap:v1", one), (None, one), ("snap:v2", two)):
        if champion is not None:
            ladder.chain.append(champion)
        assert reference.follow() == ladder.chain[-1]
        assert torch.equal(reference.log_probs(obs), actor.distribution(obs).log_probs)
    first = ["snap:v0"] if follow_first else []
    assert ladder.loads == [*first, "snap:v1", "snap:v2"]


def test_without_the_runs_champion_it_is_refused(env_spec: Any) -> None:
    section = _section({"champ": {"kind": "champion", "start": "bc"}, "bc": SNAPSHOT})
    with pytest.raises(PreflightError, match=r"0\.5\.23 or later"):
        build_references(
            msgspec.structs.replace(section, references={"champ": section.references["champ"]}),
            spec=env_spec,
            current=None,  # type: ignore[arg-type]
            build_actor=lambda device: build_model(env_spec).actor,
            codec=None,  # type: ignore[arg-type]
            device="cpu",
            printer=None,
        )


def _block(folder: Path, digest: str) -> dict[str, Any]:
    return {
        "references": {
            "bc": {"kind": "snapshot", "path": str(folder), "sha256": digest},
            "champ": {"kind": "champion", "start": "bc"},
        },
        "regularisers": [
            {
                "kind": "reference_kl",
                "name": "champ",
                "reference": "champ",
                "budget": {"kind": "constant", "value": 0.1},
                "coef": {"start": 0.0, "max": 0.0},
            }
        ],
    }


def _term(run: Any) -> ReferenceKL:
    (term,) = [t for t in run.actor_terms if isinstance(t, ReferenceKL)]
    return term


def _promote(run: Any, member: str) -> None:
    run.snapshot_store.put(member, run.model, {"step": run.cumulative_env_steps})
    run.pool.add(member, step=run.cumulative_env_steps)
    run.pool.promote(member)


def _holds(term: ReferenceKL, run: Any, member: str) -> bool:
    stored = run.snapshot_store.get(member, run.device).state_dict()
    actor = term.reference.followed.actor  # type: ignore[union-attr]
    return term.reference.held == member and all(  # type: ignore[union-attr]
        torch.equal(value, stored[name]) for name, value in actor.state_dict().items()
    )


def test_a_run_anchors_to_start_then_to_each_champion_and_resumes_holding_it(
    tmp_path: Path,
) -> None:
    """The KL is measured against ``start`` (another seed's actor) until a champion has won its
    gate, then against the champion, which is last iteration's learner and so much nearer. The
    first champion, promoted into an empty ladder without a game, moves nothing. A resumed run
    holds the champion its pool names from its first update."""
    folder = tmp_path / "other-seed"
    digest = seeded_artifact(tiny_config(tmp_path / "donor", master_seed=77), folder, coordinator)
    config = with_imitation(tiny_config(tmp_path / "run"), **_block(folder, digest))
    with coordinator(config) as run:
        run.iterate()
        term = _term(run)
        assert term.reference.held is None  # type: ignore[union-attr]
        _promote(run, "snap:c0")
        run.iterate()
        assert term.reference.held is None  # type: ignore[union-attr]
        against_start = run.rows[-1]["imitation/champ/kl"]
        _promote(run, "snap:c1")
        run.iterate()
        assert _holds(term, run, "snap:c1")
        against_champion = run.rows[-1]["imitation/champ/kl"]
        assert against_champion < against_start
        _promote(run, "snap:c2")
        run.iterate()
        assert _holds(term, run, "snap:c2")
        saved = run.checkpoint()
    with coordinator(config, resume=saved) as resumed:
        resumed.iterate()
        assert _holds(_term(resumed), resumed, "snap:c2")
