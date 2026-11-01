"""Section 19.12: ``demo_bc``, the BC cross-entropy on demonstration rows, in the actor's loss.

What is held here:

- the block decodes as a regulariser kind of its own, and its checks name what is wrong;
- steps on the term alone lower the actor's NLL of the demonstrated actions;
- a run with it moves away from one without it and publishes the term's metrics inside the
  run's schema;
- a zero coefficient leaves the run bit-identical to one without the block (the blind control);
- a shard directory whose manifest is not the one the config names is refused;
- how many batches the term has drawn survives a checkpoint;
- the run identity names the shards by digest, not by path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royaleimitate import record
from royaleimitate.config import DemoBCSpec, ImitationSection, imitation_problems
from royalelearn.errors import PreflightError

torch = pytest.importorskip("torch")

from imitation_support import with_imitation  # noqa: E402
from royalelearn.testing import coordinator, tiny_config  # noqa: E402


class _EnvOf:
    """What ``record`` needs of a learner: its config, and an environment built from it."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def build_env(self) -> Any:
        return self.config.env.factory(tuple(self.config.extra_component_modules))()


def first_legal_play(obs: dict[str, Any]) -> int:
    """A teacher a network can learn and that keeps playing: the first legal play, else the
    no-op (test_clone's)."""
    legal = np.flatnonzero(np.asarray(obs["action_mask"]))
    plays = legal[legal != 0]
    return int(plays[0]) if plays.size else 0


def _demos(config: Any, out: Path) -> tuple[Path, str]:
    directory = record(_EnvOf(config), first_legal_play, out, battles=40, seed=3)
    return directory, hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()


def _block(directory: Path, digest: str, **regulariser: Any) -> dict[str, Any]:
    return {
        "regularisers": [
            {
                "kind": "demo_bc",
                "name": "demo",
                "shards": [{"path": str(directory), "sha256": digest}],
                "batch_rows": 16,
                **regulariser,
            }
        ]
    }


def _state(run: Any) -> list[Any]:
    tensors = [t.detach().clone() for t in run.model.state_dict().values()]
    for optimizer in run.update.optimizers:
        for state in optimizer.state_dict()["state"].values():
            tensors.extend(v.clone() for v in state.values() if torch.is_tensor(v))
    return tensors


def _term(run: Any) -> Any:
    return next(t for t in run.update.extra_actor_terms if t.name == "demo")


def test_the_block_decodes_and_its_checks_name_what_is_wrong() -> None:
    section = msgspec.convert(
        {
            "regularisers": [
                {"kind": "demo_bc", "name": "demo", "shards": [{"path": "x", "sha256": "y"}]}
            ]
        },
        ImitationSection,
    )
    (spec,) = section.regularisers
    assert isinstance(spec, DemoBCSpec)
    assert (spec.batch_rows, spec.seed) == (64, 0)
    assert imitation_problems(section) == []
    bad = msgspec.convert(
        {
            "regularisers": [
                {
                    "kind": "demo_bc",
                    "name": "demo",
                    "shards": [{"path": "", "sha256": "y"}],
                    "batch_rows": 0,
                    "coef": {"kind": "constant", "value": -1.0},
                }
            ]
        },
        ImitationSection,
    )
    found = " | ".join(imitation_problems(bad))
    for words in ("needs both a path and a sha256", "batch_rows must be at least 1", "negative"):
        assert words in found, found
    empty = msgspec.convert(
        {"regularisers": [{"kind": "demo_bc", "name": "demo", "shards": []}]}, ImitationSection
    )
    assert any("names no shards" in p for p in imitation_problems(empty))


def test_the_term_alone_teaches_the_demonstrations(tmp_path: Path) -> None:
    """Steps on the term's loss alone lower the actor's NLL of the demonstrated actions: the
    gradient reaches the live actor through the rows the term draws itself."""
    from royalelearn.extensions import ActorTermInputs

    directory, digest = _demos(tiny_config(tmp_path / "demos-config"), tmp_path / "demos")
    config = with_imitation(tiny_config(tmp_path / "run"), **_block(directory, digest))
    with coordinator(config) as run:
        term = _term(run)
        actor = run.model.actor
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
        inputs = ActorTermInputs(None, None, None, None, actor, 1.0, 1.0)
        term.begin(0, run.device)
        nll = []
        for _ in range(150):
            coef, raw = term.loss(inputs, epoch=0, measure=False)
            assert coef == 0.2
            optimizer.zero_grad(set_to_none=True)
            raw.backward()
            optimizer.step()
            nll.append(float(raw.detach()))
    assert sum(nll[-20:]) < 0.5 * sum(nll[:20]), (nll[:5], nll[-5:])


def test_a_run_with_the_term_moves_and_reports_inside_its_schema(tmp_path: Path) -> None:
    from royalelearn.metrics.records import unknown_keys

    directory, digest = _demos(tiny_config(tmp_path / "demos-config"), tmp_path / "demos")

    def three(config: Any) -> tuple[list[Any], list[dict[str, Any]], Any]:
        with coordinator(config) as run:
            for _ in range(3):
                run.iterate()
            return _state(run), list(run.rows), run.schema

    plain, _rows, _schema = three(tiny_config(tmp_path / "plain"))
    taught, rows, schema = three(
        with_imitation(
            tiny_config(tmp_path / "taught"),
            **_block(directory, digest, coef={"kind": "constant", "value": 5.0}),
        )
    )
    assert not all(torch.equal(a, b) for a, b in zip(plain, taught, strict=True))
    for row in rows:
        assert row["imitation/demo/lambda"] == 5.0
        assert row["imitation/demo/demo_rows"] > 0
        assert 0.0 <= row["imitation/demo/demo_top1"] <= 1.0
        assert row["imitation/demo/demo_nll"] > 0.0
        assert not unknown_keys(row, schema)


def test_a_zero_coefficient_is_the_blind_control(tmp_path: Path) -> None:
    directory, digest = _demos(tiny_config(tmp_path / "demos-config"), tmp_path / "demos")

    def three(config: Any) -> list[Any]:
        with coordinator(config) as run:
            for _ in range(3):
                run.iterate()
            return _state(run)

    plain = three(tiny_config(tmp_path / "plain"))
    blind = three(
        with_imitation(
            tiny_config(tmp_path / "blind"),
            **_block(directory, digest, coef={"kind": "constant", "value": 0.0}),
        )
    )
    assert len(plain) == len(blind)
    assert all(torch.equal(a, b) for a, b in zip(plain, blind, strict=True))


def test_another_manifest_is_refused(tmp_path: Path) -> None:
    directory, _digest = _demos(tiny_config(tmp_path / "demos-config"), tmp_path / "demos")
    config = with_imitation(tiny_config(tmp_path / "run"), **_block(directory, "0" * 64))
    with pytest.raises(PreflightError, match="manifest sha256"), coordinator(config) as run:
        run.iterate()


def test_the_draw_count_survives_a_checkpoint(tmp_path: Path) -> None:
    directory, digest = _demos(tiny_config(tmp_path / "demos-config"), tmp_path / "demos")
    config = with_imitation(tiny_config(tmp_path / "run"), **_block(directory, digest))
    with coordinator(config) as run:
        run.iterate()
        run.iterate()
        drawn = _term(run).drawn
        assert drawn > 0
        saved = run.checkpoint()
    with coordinator(config, resume=saved) as resumed:
        assert _term(resumed).drawn == drawn
        resumed.iterate()
        assert _term(resumed).drawn > drawn


def test_the_identity_names_shards_by_digest_not_path() -> None:
    from royaleimitate.extension import EXTENSION

    one = msgspec.convert(_block(Path("a/b"), "d" * 64), ImitationSection)
    two = msgspec.convert(_block(Path("elsewhere"), "d" * 64), ImitationSection)
    other = msgspec.convert(_block(Path("a/b"), "e" * 64), ImitationSection)
    assert EXTENSION.identity_value(one) == EXTENSION.identity_value(two)
    assert EXTENSION.identity_value(one) != EXTENSION.identity_value(other)
