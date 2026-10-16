"""Under RoyaleLearn's ``ppo.policy_transform`` the learner's policy is a function of its actor's
logits, so the reference it is held to is that function of the reference's.

A snapshot reference passes its logits through the run's transform before the KL terms read them,
and the warm start's ratio guard measures the transformed policy. A field model of p(play) has
no logits to pass, and a transformed run refuses it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("royalelearn.learn.transform")

from imitation_support import seeded_artifact, with_imitation  # noqa: E402
from royaleimitate.references import SnapshotReference  # noqa: E402
from royalelearn import config as cfg  # noqa: E402
from royalelearn.extensions import MaskedCategorical, PolicyLayout, PolicyTransform  # noqa: E402
from royalelearn.testing import build_model, coordinator, tiny_config  # noqa: E402

SHARPEN = "plugin:transform_plugins:sharpen"


def _transformed(config: cfg.RunConfig) -> cfg.RunConfig:
    return msgspec.structs.replace(
        config, ppo=msgspec.structs.replace(config.ppo, policy_transform=SHARPEN)
    )


def _obs(spec: Any, rows: int = 8) -> Any:
    from royalelearn.extensions import ObsBatch

    generator = torch.Generator().manual_seed(2)
    mask = torch.rand((rows, spec.n_actions), generator=generator) < 0.3
    mask[:, 0] = True
    grid = spec.n_grid_actions
    return ObsBatch(
        spatial=torch.rand(
            (rows, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
        ),
        vector=torch.rand((rows, spec.vector_size), generator=generator),
        mask=mask,
        mask_planes=mask[:, 1:grid].float().view(rows, spec.hand_size, *spec.tiles),
    )


def test_a_snapshot_reference_reads_its_logits_through_the_transform(env_spec: Any) -> None:
    """Plant: read the actor's own distribution, and the reference is the untransformed one."""
    tiles = env_spec.tiles[0] * env_spec.tiles[1]
    layout = PolicyLayout(
        env_spec.hand_size, tiles, env_spec.n_actions - 1 - env_spec.hand_size * tiles
    )
    actor = build_model(env_spec).actor
    obs = _obs(env_spec)
    raw = MaskedCategorical(actor.logits(obs).detach(), obs.mask).log_probs
    shaped = MaskedCategorical(actor.logits(obs).detach() * 2.0, obs.mask).log_probs
    assert torch.allclose(SnapshotReference("ref", actor).log_probs(obs), raw)
    reference = SnapshotReference("ref", actor, PolicyTransform(SHARPEN, layout))
    assert torch.allclose(reference.log_probs(obs), shaped, atol=1e-6)


def test_a_run_held_to_its_own_starting_weights_starts_at_no_divergence(tmp_path: Path) -> None:
    """The init and the reference are one artifact, so before the actor moves the learner's
    policy and the reference's are one policy: the first iteration's KL is the actor's step
    alone. Plant: a reference read without the transform starts a whole temperature away."""
    folder = tmp_path / "seeded"
    digest = seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator)
    base = tiny_config(tmp_path / "run")
    config = with_imitation(
        _transformed(
            msgspec.structs.replace(base, ppo=msgspec.structs.replace(base.ppo, lr_actor=1e-9))
        ),
        init={"path": str(folder), "sha256": digest},
        references={"ref": {"kind": "snapshot", "path": str(folder), "sha256": digest}},
        regularisers=[
            {
                "kind": "reference_kl",
                "name": "bc",
                "reference": "ref",
                "budget": {"kind": "constant", "value": 0.1},
                "coef": {"start": 1.0},
            }
        ],
    )
    from royaleimitate.artifacts import read_actor_artifact
    from royaleimitate.init import loaded_ratio_guard

    rows = read_actor_artifact(folder).probe.rows  # type: ignore[union-attr]
    with coordinator(config) as run:
        run.iterate()
        kl = float(run.rows[-1]["imitation/bc/kl"])
        # The warm start's guard measures the policy the run trains: twice the logits here.
        said: list[str] = []
        guard = {"atol": 1.0, "precision": "float32", "say": said.append}
        loaded_ratio_guard(run.model, run.row_codec(), rows, **guard)
        transform, run.model.policy_transform = run.model.policy_transform, None
        loaded_ratio_guard(run.model, run.row_codec(), rows, **guard)
        run.model.policy_transform = transform
    assert 0.0 <= kl < 1e-6, kl
    shaped, raw = (float(re.search(r"\|logit\| (\S+),", line)[1]) for line in said[:2])
    # Printed to three figures.
    assert shaped == pytest.approx(2.0 * raw, rel=1e-2), said


def test_a_field_model_reference_is_refused_under_a_transform(tmp_path: Path) -> None:
    config = with_imitation(
        _transformed(tiny_config(tmp_path)),
        references={"f": {"kind": "field_mlp", "path": "f", "sha256": "0" * 64}},
        regularisers=[
            {
                "kind": "reference_kl",
                "name": "bc",
                "reference": "f",
                "factor": "noop",
                "budget": {"kind": "constant", "value": 0.1},
                "coef": {"start": 1.0},
            }
        ],
    )
    problems = cfg.check_consistency(config)
    assert any("imitation.references.f is a field model" in p for p in problems), problems
