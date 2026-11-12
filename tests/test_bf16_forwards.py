"""``autocast_dtype`` on references and on ``demo_bc``: their forwards on the bfloat16 path.

Neither enters the importance ratio, so an actor kept at float32 for the ratio guard can still
evaluate its reference, and take demo_bc's gradient, through the cheaper path. The head stays at
float32 either way, and the live actor's own forwards keep the run's precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from imitation_support import seeded_artifact, with_imitation  # noqa: E402
from royaleimitate.config import (  # noqa: E402
    ChampionReferenceSpec,
    DemoBCSpec,
    ImitationSection,
    SnapshotReferenceSpec,
    imitation_problems,
)
from royaleimitate.references import actor_precision, set_precision  # noqa: E402
from royalelearn.testing import build_model, coordinator, tiny_config  # noqa: E402
from test_demo_bc import _block as _demo_block  # noqa: E402
from test_demo_bc import _demos, _term  # noqa: E402
from test_policy_transform import _obs  # noqa: E402


def test_unset_nothing_is_encoded() -> None:
    for spec in (
        SnapshotReferenceSpec(path="x", sha256="y"),
        ChampionReferenceSpec(start="bc"),
        DemoBCSpec(name="demo", shards=[]),
    ):
        assert b"autocast_dtype" not in msgspec.json.encode(spec)


@pytest.mark.parametrize("value", ["float16", "bf16"])
def test_a_precision_nothing_can_run_is_named(value: str) -> None:
    section = msgspec.convert(
        {
            "references": {
                "bc": {"kind": "snapshot", "path": "x", "sha256": "0" * 64, "autocast_dtype": value}
            },
            "regularisers": [
                {
                    "kind": "demo_bc",
                    "name": "demo",
                    "shards": [{"path": "x", "sha256": "y"}],
                    "autocast_dtype": value,
                },
                {"kind": "reference_kl", "name": "kl", "reference": "bc"},
            ],
        },
        type=ImitationSection,
    )
    found = [p for p in imitation_problems(section) if f"autocast_dtype is {value!r}" in p]
    assert len(found) == 2


def test_the_trunk_takes_the_precision_and_the_head_stays_float32(env_spec: Any) -> None:
    """Plant: leave the head under the autocast, and its input is bfloat16."""
    actor = build_model(env_spec).actor
    seen: dict[str, Any] = {}
    actor.trunk.register_forward_hook(lambda _m, _i, out: seen.__setitem__("trunk", out.dtype))
    actor.head.register_forward_pre_hook(
        lambda _m, args: seen.__setitem__("head", args[0].dtype)
    )
    obs = _obs(env_spec, rows=6)
    with torch.no_grad():
        exact = actor.logits(obs)
        with actor_precision(actor, "bfloat16"):
            fast = actor.logits(obs)
            assert seen == {"trunk": torch.bfloat16, "head": torch.float32}
        assert (actor.autocast_dtype, actor.head_float32) == (torch.float32, False)
        again = actor.logits(obs)
    assert fast.dtype == torch.float32
    assert torch.equal(again, exact)
    assert not torch.equal(fast, exact)
    assert torch.allclose(fast, exact, atol=0.1)
    set_precision(actor, msgspec.UNSET)
    assert actor.autocast_dtype == torch.float32


def _reference_run(tmp_path: Path, dtype: Any) -> Any:
    folder = tmp_path / "other-seed"
    digest = seeded_artifact(tiny_config(tmp_path / "donor", master_seed=77), folder, coordinator)
    reference: dict[str, Any] = {"kind": "snapshot", "path": str(folder), "sha256": digest}
    if dtype is not None:
        reference["autocast_dtype"] = dtype
    return with_imitation(
        tiny_config(tmp_path / "run"),
        references={"bc": reference},
        regularisers=[
            {
                "kind": "reference_kl",
                "name": "bc",
                "reference": "bc",
                "coef": {"start": 0.0, "max": 0.0},
            }
        ],
    )


def test_a_reference_at_bfloat16_passes_its_self_test_and_measures_the_same_kl(
    tmp_path: Path,
) -> None:
    """The self-test runs before the precision is set, so it still checks the artifact; the KL
    it then measures is the float32 reference's, within bfloat16's rounding."""
    kls = {}
    for dtype in (None, "bfloat16"):
        with coordinator(_reference_run(tmp_path / str(dtype), dtype)) as run:
            (term,) = [t for t in run.actor_terms if getattr(t, "name", "") == "bc"]
            assert term.reference.actor.head_float32 is (dtype is not None)
            run.iterate()
            kls[dtype] = run.rows[-1]["imitation/bc/kl"]
            # The live actor is untouched.
            assert run.model.actor.autocast_dtype == torch.float32
    assert kls[None] > 0.0
    assert kls["bfloat16"] == pytest.approx(kls[None], rel=0.05)


def test_demo_bc_at_bfloat16_trains_and_leaves_the_actor_as_it_was(tmp_path: Path) -> None:
    directory, digest = _demos(tiny_config(tmp_path / "demos-config"), tmp_path / "demos")
    config = with_imitation(
        tiny_config(tmp_path / "taught"),
        **_demo_block(
            directory, digest, coef={"kind": "constant", "value": 5.0}, autocast_dtype="bfloat16"
        ),
    )
    with coordinator(config) as run:
        assert _term(run).autocast_dtype == "bfloat16"
        before = [p.detach().clone() for p in run.model.actor.parameters()]
        run.iterate()
        row = run.rows[-1]
        assert row["imitation/demo/demo_nll"] > 0.0
        assert row["ppo/ratio_max_abs_dev"] == 0.0
        assert (run.model.actor.autocast_dtype, run.model.actor.head_float32) == (
            torch.float32,
            False,
        )
        assert any(
            not torch.equal(a, b) for a, b in zip(before, run.model.actor.parameters(), strict=True)
        )


#: Runs bfloat16 on the processor, so conftest runs it without oneDNN (see there why).
CPU_BFLOAT16 = True
