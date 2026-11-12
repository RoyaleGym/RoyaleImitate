"""The two config sections of learning from demonstrations (section 19), and their checks.

``warm_start`` starts the actor from saved weights and schedules its learning-rate scale, which
is how a run holds the actor still while the critic catches up. ``imitation`` anchors the policy
to reference policies with the reference-KL regularisers. Each is a top-level config key owned by
an extension (``royalelearn.extensions``), and each carries its own alarm thresholds, which stay
out of the run identity as the core's do.
"""

from __future__ import annotations

import msgspec

from royalelearn.extensions import ConstantSpec, PiecewiseConstantSpec, ScheduleSpec

__all__ = [
    "KL_FACTORS",
    "ROW_OPS",
    "ChampionReferenceSpec",
    "CoefSpec",
    "DemoBCSpec",
    "DemoShards",
    "FieldMLPReferenceSpec",
    "ImitationAlarms",
    "ImitationSection",
    "InitSpec",
    "ReferenceKLSpec",
    "ReferenceSpec",
    "RegulariserSpec",
    "RowCondition",
    "SnapshotReferenceSpec",
    "WarmStartAlarms",
    "WarmStartSection",
    "imitation_problems",
    "schedule_values",
    "warm_start_problems",
]

#: The two ways a reference-KL regulariser can compare policies (section 19.7).
KL_FACTORS = ("joint", "noop_marginal")
#: The comparisons an ``exclude_when`` condition can make.
ROW_OPS = ("<", "<=", ">", ">=")


class InitSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """The actor's starting weights: an actor artifact and its digest (section 19.4)."""

    path: str
    #: The artifact digest of the folder (section 19.2), not a hash of the weights file alone.
    sha256: str
    #: How far the loaded actor's log-probabilities on the artifact's probe rows may be from the
    #: ones recorded when it was written.
    self_test_atol: float = 1e-5


class WarmStartAlarms(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """Thresholds of the two alarms about letting a frozen actor go (section 19.9)."""

    handoff_window: int = 20
    handoff_kl: float = 0.05
    handoff_clip: float = 0.3
    ev_at_unfreeze: float = 0.3


class WarmStartSection(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """``warm_start``: where the actor starts, and how fast it may move at first."""

    init: InitSpec | None = None
    #: Multiplies the actor's learning rate after the backoff has set it. Zero freezes the actor.
    actor_lr_scale: ScheduleSpec | None = None
    alarms: WarmStartAlarms = WarmStartAlarms()


class SnapshotReferenceSpec(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    kw_only=True,
    tag_field="kind",
    tag="snapshot",
):
    """A reference policy that is an actor artifact, run on a second copy of the run's actor."""

    path: str
    sha256: str
    #: How far the reference's log-probabilities on the artifact's probe rows may be from the ones
    #: recorded when it was written, as ``warm_start.init.self_test_atol`` is for the init. Unset
    #: means 1e-5 and keeps a run's identity as it was. A run whose kernels differ from the
    #: recording's (``determinism.tier`` throughput picks its convolutions by timing) states it.
    self_test_atol: float | msgspec.UnsetType = msgspec.UNSET
    #: The forward at this precision, the head at float32, whatever the run's actor runs at.
    #: A reference never enters the importance ratio, so an actor kept at float32 for the ratio
    #: guard can be compared with a reference evaluated on the bfloat16 path. "bfloat16" or
    #: "float32"; unset means the precision the run's actor has, and is left out of the
    #: encoding.
    autocast_dtype: str | msgspec.UnsetType = msgspec.UNSET


class FieldMLPReferenceSpec(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    kw_only=True,
    tag_field="kind",
    tag="field_mlp",
):
    """A reference over named observation fields that says only how likely a play is."""

    path: str
    sha256: str


class ChampionReferenceSpec(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    kw_only=True,
    tag_field="kind",
    tag="champion",
):
    """The ladder's champion, on a second copy of the run's actor, followed from one promotion
    to the next: its weights are loaded again at the first update after a gate promotes another.
    Until there is a champion to follow it is ``start``."""

    #: A ``snapshot`` reference of this section, the anchor until there is a champion to follow.
    start: str
    #: The champion's forward at this precision, the head at float32, whatever the run's actor
    #: runs at (``start`` has its own). A reference never enters the importance ratio, so an
    #: actor kept at float32 for the ratio guard can be compared with a reference evaluated on
    #: the bfloat16 path. "bfloat16" or "float32"; unset means the precision the run's actor
    #: has, and is left out of the encoding.
    autocast_dtype: str | msgspec.UnsetType = msgspec.UNSET
    #: The run's first champion is promoted into an empty ladder without a game. False: it is
    #: not followed, and ``start`` anchors until a champion has won its gate by playing. True:
    #: it is followed as every later one is.
    follow_first: bool = False


ReferenceSpec = SnapshotReferenceSpec | FieldMLPReferenceSpec | ChampionReferenceSpec


class RowCondition(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """One element of a named vector field compared with a value (section 19.7)."""

    field: str
    index: int = 0
    op: str
    value: float


class CoefSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """The adaptive coefficient of section 19.8: where it starts and how it may move."""

    start: float = 1.0
    max: float = 10.0
    #: A floor that may fall over the run: high while the anchor is protected, then low.
    min: ScheduleSpec = ConstantSpec(0.0)
    up: float = 1.5
    down: float = 1.5
    #: The dead band around the budget: no move while ``budget/band <= kl <= band*budget``.
    band: float = 1.5


class ReferenceKLSpec(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    kw_only=True,
    tag_field="kind",
    tag="reference_kl",
):
    """KL(reference || policy) on the choice rows, held to a budget (sections 19.7-19.8)."""

    name: str
    reference: str
    factor: str = "joint"
    exclude_when: list[RowCondition] = []
    #: The KL the coefficient holds the policy to, in nats per choice row.
    budget: ScheduleSpec = ConstantSpec(0.1)
    coef: CoefSpec = CoefSpec()


class DemoShards(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """One demonstration shard directory, named by the sha256 of its manifest (section 19.12)."""

    path: str
    sha256: str


class DemoBCSpec(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    kw_only=True,
    tag_field="kind",
    tag="demo_bc",
):
    """The BC cross-entropy on a batch of demonstration rows, added to the actor's loss each
    minibatch with a scheduled coefficient (section 19.12).

    Each minibatch draws ``batch_rows`` rows from the shards' learning split, in an order set by
    ``seed``, and adds ``coef * mean NLL`` scaled by the minibatch's share of its ppo batch, so each
    optimizer step adds ``coef`` times one mean NLL (over ``batch_size / minibatch_size`` draws),
    and
    an iteration ``n_epochs * samples / batch_size`` of them. The NLL
    is the actor's own masked distribution, as ``clone`` trains it, not ``ppo.policy_transform``'s.
    """

    name: str
    shards: list[DemoShards]
    coef: ScheduleSpec = ConstantSpec(0.2)
    batch_rows: int = 64
    seed: int = 0
    #: The live actor's forward on the demonstration rows at this precision, its head at
    #: float32, whatever the actor runs at elsewhere. The BC loss never enters the importance
    #: ratio, so an actor kept at float32 for the ratio guard can take this term's gradient
    #: through the bfloat16 path; the parameters stay float32. "bfloat16" or "float32"; unset
    #: means the actor's own precision, and is left out of the encoding.
    autocast_dtype: str | msgspec.UnsetType = msgspec.UNSET


RegulariserSpec = ReferenceKLSpec | DemoBCSpec


class ImitationAlarms(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """Thresholds of the regularisers' two alarms (section 19.9)."""

    ref_kl_warn: float = 1.0
    lambda_saturated_patience: int = 10


class ImitationSection(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """``imitation``: the reference policies and the regularisers that anchor the policy to them."""

    references: dict[str, ReferenceSpec] = {}
    regularisers: list[RegulariserSpec] = []
    alarms: ImitationAlarms = ImitationAlarms()


def schedule_values(spec: ScheduleSpec) -> list[float]:
    """Every value a schedule can take, for a bound that has to hold over the whole run."""
    if isinstance(spec, ConstantSpec):
        return [spec.value]
    if isinstance(spec, PiecewiseConstantSpec):
        return [value for _, value in spec.points]
    return [spec.start, spec.end]


def warm_start_problems(section: WarmStartSection) -> list[str]:
    """What is wrong with a ``warm_start`` section that can be seen without building anything."""
    problems: list[str] = []
    if section.init is not None and not section.init.self_test_atol >= 0.0:
        problems.append("warm_start.init.self_test_atol must be at least 0")
    if section.actor_lr_scale is not None and any(
        not value >= 0.0 for value in schedule_values(section.actor_lr_scale)
    ):
        problems.append("warm_start.actor_lr_scale takes a negative value")
    alarms = section.alarms
    if alarms.handoff_window < 0:
        problems.append("warm_start.alarms.handoff_window must be at least 0")
    return problems


#: The precisions a reference's or demo_bc's forward may take.
PRECISIONS = ("bfloat16", "float32")


def _precision_problems(value: object, label: str) -> list[str]:
    if value is msgspec.UNSET or value in PRECISIONS:
        return []
    return [f"{label}.autocast_dtype is {value!r}, which is not one of {', '.join(PRECISIONS)}"]


def imitation_problems(section: ImitationSection) -> list[str]:
    """What is wrong with an ``imitation`` section that can be seen without building anything.

    Fields that need the environment -- an ``exclude_when`` naming a vector field -- are checked
    at preflight, where the layout is known.
    """
    problems: list[str] = []
    where = "imitation"
    for name, reference in section.references.items():
        if not name or "/" in name:
            problems.append(f"{where}.references: {name!r} is not a usable name")
        problems.extend(
            _precision_problems(
                getattr(reference, "autocast_dtype", msgspec.UNSET), f"{where}.references.{name}"
            )
        )
        if isinstance(reference, ChampionReferenceSpec):
            start = section.references.get(reference.start)
            if not isinstance(start, SnapshotReferenceSpec):
                problems.append(
                    f"{where}.references.{name}: start {reference.start!r} must name a snapshot "
                    "reference of this section, the anchor until the ladder has a champion"
                )
            continue
        if not reference.path or not reference.sha256:
            problems.append(f"{where}.references.{name} needs both a path and a sha256")
        atol = getattr(reference, "self_test_atol", msgspec.UNSET)
        if atol is not msgspec.UNSET and not atol >= 0.0:
            problems.append(f"{where}.references.{name}.self_test_atol must be at least 0")
    seen: set[str] = set()
    for index, reg in enumerate(section.regularisers):
        label = f"{where}.regularisers[{index}] ({reg.name!r})"
        if not reg.name or "/" in reg.name or reg.name != reg.name.strip():
            problems.append(f"{label}: the name must be one metric segment")
        if reg.name in seen:
            problems.append(f"{label}: two regularisers are named {reg.name!r}")
        seen.add(reg.name)
        if isinstance(reg, DemoBCSpec):
            problems.extend(_demo_bc_problems(reg, label))
            continue
        reference = section.references.get(reg.reference)
        if reference is None:
            declared = ", ".join(sorted(section.references)) or "none"
            problems.append(
                f"{label}: reference {reg.reference!r} is not declared (declared: {declared})"
            )
        if reg.factor not in KL_FACTORS:
            problems.append(f"{label}: factor {reg.factor!r} is not one of {', '.join(KL_FACTORS)}")
        elif reg.factor == "joint" and isinstance(reference, FieldMLPReferenceSpec):
            problems.append(
                f"{label}: a field_mlp reference says only how likely a play is, so it can back "
                "factor 'noop_marginal' and not 'joint'"
            )
        for condition in reg.exclude_when:
            if condition.op not in ROW_OPS:
                problems.append(
                    f"{label}: exclude_when op {condition.op!r} is not one of {', '.join(ROW_OPS)}"
                )
            if condition.index < 0:
                problems.append(f"{label}: exclude_when index {condition.index} is negative")
        if any(not value >= 0.0 for value in schedule_values(reg.budget)):
            problems.append(f"{label}: the budget takes a negative value")
        coef = reg.coef
        floors = schedule_values(coef.min)
        if any(not value >= 0.0 for value in (coef.start, coef.max, *floors)):
            problems.append(f"{label}: a coefficient (start, max or min) is negative")
        elif coef.max < max(floors):
            problems.append(
                f"{label}: coef.max {coef.max:g} is below the largest floor coef.min takes, "
                f"{max(floors):g}"
            )
        for knob in ("up", "down", "band"):
            if not getattr(coef, knob) > 1.0:
                problems.append(f"{label}: coef.{knob} must be above 1")
    return problems


def _demo_bc_problems(reg: DemoBCSpec, label: str) -> list[str]:
    problems: list[str] = []
    if not reg.shards:
        problems.append(f"{label}: demo_bc names no shards")
    for index, shard in enumerate(reg.shards):
        if not shard.path or not shard.sha256:
            problems.append(f"{label}: shards[{index}] needs both a path and a sha256")
    if reg.batch_rows < 1:
        problems.append(f"{label}: batch_rows must be at least 1")
    if reg.seed < 0:
        problems.append(f"{label}: seed must be at least 0")
    problems.extend(_precision_problems(reg.autocast_dtype, label))
    if any(not value >= 0.0 for value in schedule_values(reg.coef)):
        problems.append(f"{label}: the coefficient takes a negative value")
    return problems
