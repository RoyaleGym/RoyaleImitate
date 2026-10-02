"""A teacher's battles as demonstrations, and a policy trained to copy them.

``record`` plays a teacher -- a scripted bot by name, an ``Opponent``, or any policy such as a
learner's saved bot -- on both seats of battles in a ``royalelearn.Learner``'s own environment, and
stores every decision it makes as a shard row (``royaleimitate.shards``), one group per battle.

``clone`` trains that learner's own network to copy those decisions: the cross-entropy of the
teacher's action under the masked distribution, weighted by each row's weight, with AdamW, a
cosine learning-rate schedule and early stopping on the validation rows (the by-battle split of
``royaleimitate.split``). It writes an actor folder with probe rows, which a run's ``warm_start``
loads, and returns its digest. Plain behaviour cloning at ordinary defaults: nothing here is tuned.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from royalelearn.extensions import PreflightError, write_policy_record

from .artifacts import ProbeSet, artifact_digest, probe_log_probs, write_actor_artifact
from .shards import ShardContext, ShardReader, ShardWriter

__all__ = ["PROBE_ROWS", "clone", "record"]

#: How many validation rows the clone's folder keeps as probe rows.
PROBE_ROWS = 1024
#: The scripted bots a teacher may be named by, besides "random" and "noop".
_SCRIPTED = ("first_affordable", "defend", "push", "patient")


def _teacher(teacher: Any) -> Any:
    """The teacher as something with ``act(obs, mask, rng) -> int``."""
    if isinstance(teacher, str):
        from royalelearn.extensions import build_opponent

        name = "random_legal" if teacher == "random" else teacher
        if name in ("noop", "random_legal", *_SCRIPTED):
            return build_opponent(name)
        raise PreflightError(
            f"teacher {teacher!r} is not a policy, an Opponent or one of the scripted bots: "
            f"random, noop, {', '.join(_SCRIPTED)}"
        )
    if hasattr(teacher, "act"):
        return teacher
    if callable(teacher):
        policy: Callable[[Any], int] = teacher

        class _Policy:
            def act(self, obs: Any, mask: Any, rng: Any) -> int:
                return int(policy(obs))

        return _Policy()
    raise PreflightError(f"teacher {teacher!r} is not a policy, an Opponent or a name")


def record(
    learner: Any,
    teacher: Any,
    out: str | Path,
    *,
    battles: int = 100,
    seed: int = 0,
) -> Path:
    """Play ``teacher`` on both seats of ``battles`` battles of ``learner``'s environment and store
    every decision as a shard row. Returns the shard directory (under ``out``) to give ``clone``.

    ``teacher`` is a policy (a function from one seat's observation to an action, as
    ``Learner.load_policy`` returns), a ``royalegym`` ``Opponent``, or one of the scripted bots by
    name: "random", "noop", "first_affordable", "defend", "push" or "patient".
    """
    player = _teacher(teacher)
    config = learner.config
    context = ShardContext.of_config(config)
    rng = np.random.default_rng(seed)
    producer = {"tool": "royaleimitate.record", "battles": int(battles), "seed": int(seed)}
    with ShardWriter(out, context, producer=producer) as writer:
        for battle in range(int(battles)):
            env = learner.build_env()
            try:
                obs, _info = env.reset(seed=int(seed) * 1_000_003 + battle)
                step = 0
                while env.agents:
                    actions = {}
                    for seat, agent in enumerate(sorted(env.agents)):
                        mask = np.asarray(obs[agent]["action_mask"])
                        action = int(player.act(obs[agent], mask, rng))
                        actions[agent] = action
                        writer.add(
                            obs[agent],
                            action=action,
                            weight=1.0,
                            group=int(seed) * 1_000_003 + battle,
                            seat=0 if agent == "blue" else 1 if agent == "red" else seat,
                            tick=step,
                        )
                    obs, _rewards, _terminated, _truncated, _info = env.step(actions)
                    step += 1
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
    return Path(out) / context.engine_key


def clone(
    learner: Any,
    demonstrations: str | Path,
    out: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 3,
    seed: int = 0,
) -> str:
    """Train ``learner``'s network to copy the actions in ``demonstrations`` (a directory
    ``record`` returned) and write it to the folder ``out``. Returns the folder's digest, the
    ``sha256`` a ``warm_start.init`` names it by.

    Stops early once the validation NLL has not improved for ``patience`` epochs, and keeps the
    weights of the best epoch. The folder's ``spec.json`` records the validation NLL before
    training and after each epoch.
    """
    import torch

    from royalelearn.extensions import LearningCoordinator

    config = learner.config
    reader = ShardReader(demonstrations, ShardContext.of_config(config))
    if not reader.manifest.validation_rows:
        raise PreflightError(
            f"{demonstrations} has no row in the validation split, which holds out about one "
            "battle in twenty: record 100 battles or more"
        )
    kwargs = dict(getattr(learner, "_kwargs", {}))
    kwargs["printer"] = lambda _line: None
    kwargs["install_signal_handler"] = False
    with tempfile.TemporaryDirectory() as scratch:
        run = LearningCoordinator(
            config, device=learner.device, resume=None, run_dir=Path(scratch) / "run", **kwargs
        )
        with run:
            torch.manual_seed(int(seed))
            codec = run.row_codec()
            actor = run.model.actor
            optimizer = torch.optim.AdamW(
                actor.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

            def nll(split: str, epoch: int, *, train: bool) -> float:
                total, weight = 0.0, 0.0
                for batch in reader.batches(
                    codec, split=split, batch_rows=batch_size, seed=seed, epoch=epoch
                ):
                    with torch.set_grad_enabled(train):
                        distribution = actor.distribution(batch.obs)
                        log_probs = distribution.log_prob(batch.actions)
                        loss = -(log_probs * batch.weights).sum() / batch.weights.sum().clamp_min(
                            1e-8
                        )
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach()) * float(batch.weights.sum())
                    weight += float(batch.weights.sum())
                return total / max(weight, 1e-8)

            history = [nll("validation", 0, train=False)]
            best, best_state, waited = history[0], copy.deepcopy(actor.state_dict()), 0
            for epoch in range(int(epochs)):
                nll("train", epoch, train=True)
                schedule.step()
                history.append(nll("validation", 0, train=False))
                if history[-1] < best:
                    best, best_state, waited = history[-1], copy.deepcopy(actor.state_dict()), 0
                else:
                    waited += 1
                    if waited >= patience:
                        break
            actor.load_state_dict(best_state)
            actor.eval()

            rows: list[np.ndarray] = []
            for index in range(len(reader.manifest.parts)):
                packed, _scalars = reader.packed(index, codec, split="validation")
                rows.append(packed)
                if sum(r.shape[0] for r in rows) >= PROBE_ROWS:
                    break
            probe_rows = np.concatenate(rows)[:PROBE_ROWS]
            log_probs, _mask = probe_log_probs(actor, codec, probe_rows)
            import msgspec

            template = run.snapshot_template
            spec = msgspec.structs.replace(
                template,
                meta={
                    **template.meta,
                    "clone": {
                        "validation_nll": [round(v, 6) for v in history],
                        "best_validation_nll": round(best, 6),
                        "demonstration_rows": int(reader.manifest.rows),
                        "engine_key": reader.manifest.engine_key,
                    },
                },
            )
            write_actor_artifact(
                out,
                actor.state_dict(),
                spec,
                probe=ProbeSet(rows=probe_rows, log_probs=log_probs),
            )
            # So that ``Learner.load_policy(out)`` plays the clone, as well as a warm start.
            write_policy_record(out, run.spec, config.net)
    return artifact_digest(out)
