"""The sections' alarms and metric keys: what each contributes to a run, and only when the run can
feed it.

The regularisers' two alarms read families named by the run's own regularisers; the freeze's two
read the core's ``ppo/`` freeze keys and are RoyaleLearn's, brought by ``warm_start`` with its
thresholds.
"""

from __future__ import annotations

from typing import Any

import pytest

from royaleimitate.schema import IMITATION_PATTERNS
from royalelearn import config as cfg
from royalelearn.extensions import extension_alarms, schema_contributions, with_sections
from royalelearn.metrics import schema
from royalelearn.metrics.schema import SchemaContribution, for_run

IMITATION: dict[str, Any] = {
    "references": {"ref": {"kind": "snapshot", "path": "ref", "sha256": "0" * 64}},
    "regularisers": [
        {
            "kind": "reference_kl",
            "name": "bc",
            "reference": "ref",
            "budget": {"kind": "constant", "value": 0.1},
            "coef": {"start": 1.0},
        }
    ],
}

#: The regularisers' families, published on a run with regularisers only.
PUBLISHED_PATTERNS = (
    "imitation/{name}/budget",
    "imitation/{name}/grad_ratio",
    "imitation/{name}/kl",
    "imitation/{name}/kl_card",
    "imitation/{name}/kl_noop",
    "imitation/{name}/kl_tile",
    "imitation/{name}/lambda",
    "imitation/{name}/lambda_at_max",
    "imitation/{name}/ref_p_noop",
    "imitation/{name}/rows_frac",
    "imitation/{name}/top1_agree",
)


def _alarm(config: cfg.RunConfig, name: str) -> Any:
    (found,) = [alarm for alarm in extension_alarms(config) if alarm.name == name]
    return found


def test_the_families_are_the_ones_written_down() -> None:
    live = {pattern.template for pattern in IMITATION_PATTERNS}
    assert live == set(PUBLISHED_PATTERNS)
    assert not live & {pattern.template for pattern in schema.PATTERNS}


def test_a_run_with_regularisers_knows_their_keys_and_holds_their_alarms() -> None:
    config = with_sections(cfg.RunConfig(), imitation=IMITATION)
    run_schema = for_run(schema_contributions(config))
    assert run_schema.is_known("imitation/bc/kl")
    names = [alarm.name for alarm in extension_alarms(config)]
    assert names == ["imitation_ref_kl_high", "imitation_lambda_saturated"]


def test_a_section_contributes_only_what_its_run_can_feed() -> None:
    """An init without a schedule has no freeze to watch; references without regularisers have
    no KL to report."""
    init_only = with_sections(cfg.RunConfig(), warm_start={"init": {"path": "x", "sha256": "y"}})
    references_only = with_sections(
        cfg.RunConfig(), imitation={"references": IMITATION["references"]}
    )
    for config in (init_only, references_only):
        assert extension_alarms(config) == ()
        assert schema_contributions(config) == (SchemaContribution(),)


def test_the_family_alarms_fire_on_their_thresholds() -> None:
    config = with_sections(
        cfg.RunConfig(),
        imitation={**IMITATION, "alarms": {"ref_kl_warn": 0.5, "lambda_saturated_patience": 3}},
    )
    kl_high = _alarm(config, "imitation_ref_kl_high")
    assert kl_high.holds({"imitation/bc/kl": 0.51})
    assert not kl_high.holds({"imitation/bc/kl": 0.49})
    assert not kl_high.holds({}), "a row with no member is not a firing"
    saturated = _alarm(config, "imitation_lambda_saturated")
    assert saturated.patience == 3
    assert saturated.holds({"imitation/bc/lambda_at_max": 1.0})
    row = {"imitation/bc/kl": 1.5, "imitation/timing/kl": 0.2}
    assert "imitation/bc/kl" in kl_high.message(row)
    assert "imitation/timing/kl" not in kl_high.message(row)


@pytest.mark.parametrize(
    ("alarm", "keys", "over", "under"),
    [
        (
            "actor_handoff",
            ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
            (0.071, 0.0, 5.0),
            (0.069, 0.0, 5.0),
        ),
        (
            "actor_handoff",
            ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
            (0.0, 0.41, 5.0),
            (0.0, 0.39, 5.0),
        ),
        (
            "actor_handoff",
            ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
            (0.2, 0.0, 5.0),
            (0.2, 0.0, 6.0),
        ),
        ("critic_unready", ("ppo/ev_at_unfreeze",), (0.19,), (0.21,)),
    ],
)
def test_each_freeze_threshold_is_the_one_warm_start_names(
    alarm: str, keys: tuple[str, ...], over: tuple[float, ...], under: tuple[float, ...]
) -> None:
    """Four distinct thresholds, each probed either side of its own number."""
    config = with_sections(
        cfg.RunConfig(),
        warm_start={
            "actor_lr_scale": {"kind": "constant", "value": 1.0},
            "alarms": {
                "handoff_window": 5,
                "handoff_kl": 0.07,
                "handoff_clip": 0.4,
                "ev_at_unfreeze": 0.2,
            },
        },
    )
    built = _alarm(config, alarm)
    assert built.holds(dict(zip(keys, over, strict=True)))
    assert not built.holds(dict(zip(keys, under, strict=True)))
