"""Imitation learning for RoyaleLearn, as an optional add-on.

Two config sections, found by RoyaleLearn through this package's entry points:

- ``warm_start``: start the actor from saved weights, check them against their own probe rows,
  and hold the actor still at first while the critic catches up;
- ``imitation``: anchor the policy to reference policies with the reference-KL regulariser and
  its adaptive coefficient.

And the tools that make what they read: saved-policy folders (``artifacts``), field models
(``field_model``, ``fit``), and demonstration shards (``shards``, ``split``). And ``public_log``:
the observation's fair fields from a timed log of card plays, with no engine running.
And ``from_replays``: human games from the public IL_Replay dataset, as demonstration shards.
"""

from __future__ import annotations

__version__ = "0.2.12"

__all__ = ["clone", "from_replays", "record", "save_actor"]


def save_actor(learner: object, folder: object, *, seed: int = 0) -> str:
    """Write a trained ``royalelearn.Learner``'s actor as an actor artifact; returns its digest.

    See ``royaleimitate.artifacts.save_actor``. Imported on first use, so importing this package
    does not import torch.
    """
    from .artifacts import save_actor as save

    return save(learner, folder, seed=seed)  # type: ignore[arg-type]


def record(learner: object, teacher: object, out: object, **kwargs: object) -> object:
    """A teacher's battles in ``learner``'s environment, as shard rows; see
    ``royaleimitate.cloning.record``."""
    from .cloning import record as run

    return run(learner, teacher, out, **kwargs)  # type: ignore[arg-type]


def clone(learner: object, demonstrations: object, out: object, **kwargs: object) -> str:
    """``learner``'s network trained to copy ``demonstrations``, written to ``out``; see
    ``royaleimitate.cloning.clone``."""
    from .cloning import clone as run

    return run(learner, demonstrations, out, **kwargs)  # type: ignore[arg-type]


def from_replays(learner: object, out: object, **kwargs: object) -> object:
    """Human games from the IL_Replay dataset, replayed in ``learner``'s environment, as shard
    rows; see ``royaleimitate.replays.from_replays``."""
    from .replays import from_replays as run

    return run(learner, out, **kwargs)  # type: ignore[arg-type]
