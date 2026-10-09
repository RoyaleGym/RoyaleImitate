"""Add ``actor_digest`` to an actor folder written before it existed, every tensor left as it was.

A warm start loads an actor folder into a run whose architecture agrees with it. Before
royalelearn 0.5.16, "agrees" meant the whole ``arch_digest``: the precision and the device the
folder was trained at included, and the critic it does not carry. So a bfloat16 student could
not start from a float32 clone. A folder that states ``actor_digest`` (what its actor's weights
compute) is held to that instead, and folders written from royalelearn 0.5.16 and royaleimitate
0.2.12 on state it. This adds it to one written before.

It is read from the folder's own ``policy.json``, and only after that file is shown to describe
these weights: its network and environment must give the ``arch_digest`` the folder was saved
with. Only ``spec.json`` is rewritten, so the folder's sha256 -- what a ``warm_start.init``
names it by -- moves, and the new one is printed beside the old::

    python -m royaleimitate.stamp runs/my_clone
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import NamedTuple

import msgspec

from royalelearn.extensions import (
    SPEC_NAME,
    WEIGHTS_NAME,
    EnvSpec,
    NetConfig,
    PreflightError,
    SnapshotSpec,
    actor_digest_of,
    arch_digest_of,
)

from .artifacts import artifact_digest

__all__ = ["POLICY_NAME", "Stamped", "folder_actor_digest", "main", "stamp_actor_digest"]

#: What ``write_policy_record`` writes beside an actor's weights.
POLICY_NAME = "policy.json"


class Stamped(NamedTuple):
    """What a stamp did: the folder's sha256 before and after, and the digest it now states."""

    folder: Path
    old_sha256: str
    new_sha256: str
    actor_digest: str


def folder_actor_digest(folder: str | os.PathLike[str], spec: SnapshotSpec) -> str | None:
    """The ``actor_digest`` of ``folder``'s weights, read from its ``policy.json``; None when it
    has none, or when that file does not give the ``arch_digest`` the weights were saved with."""
    root = Path(folder)
    if not (root / POLICY_NAME).is_file():
        return None
    record = msgspec.json.decode((root / POLICY_NAME).read_bytes())
    env_spec = msgspec.convert(record["env_spec"], EnvSpec)
    net = msgspec.convert(record["net"], NetConfig)
    if arch_digest_of(env_spec, net) != spec.arch_digest:
        return None
    return actor_digest_of(env_spec, net)


def stamp_actor_digest(folder: str | os.PathLike[str]) -> Stamped:
    """Write ``actor_digest`` into ``folder``'s ``spec.json``. A folder that already states the
    right one is left alone; one whose ``policy.json`` does not describe its weights is refused."""
    root = Path(folder)
    for name in (SPEC_NAME, WEIGHTS_NAME, POLICY_NAME):
        if not (root / name).is_file():
            raise PreflightError(
                f"{root} has no {name}: an actor folder needs its {SPEC_NAME}, its "
                f"{WEIGHTS_NAME} and the {POLICY_NAME} that says which network they are"
            )
    before = artifact_digest(root)
    spec = msgspec.json.decode((root / SPEC_NAME).read_bytes(), type=SnapshotSpec)
    record = msgspec.json.decode((root / POLICY_NAME).read_bytes())
    env_spec = msgspec.convert(record["env_spec"], EnvSpec)
    net = msgspec.convert(record["net"], NetConfig)
    said = arch_digest_of(env_spec, net)
    if said != spec.arch_digest:
        raise PreflightError(
            f"{root}: its {POLICY_NAME} describes arch_digest {said}, and its weights were saved "
            f"from {spec.arch_digest}. Not stamped: the digest would describe another network"
        )
    digest = actor_digest_of(env_spec, net)
    if spec.actor_digest is not msgspec.UNSET:
        if spec.actor_digest != digest:
            raise PreflightError(
                f"{root} states actor_digest {spec.actor_digest}, and its {POLICY_NAME} gives "
                f"{digest}. Left as it is"
            )
        return Stamped(root, before, before, digest)

    weights = (root / WEIGHTS_NAME).read_bytes()
    tensors = _tensors(weights)
    stamped = msgspec.structs.replace(spec, actor_digest=digest)
    temporary = root / f"{SPEC_NAME}.stamping"
    temporary.write_bytes(msgspec.json.encode(stamped))
    os.replace(temporary, root / SPEC_NAME)

    after = (root / WEIGHTS_NAME).read_bytes()
    if hashlib.sha256(after).digest() != hashlib.sha256(weights).digest():
        raise AssertionError(f"{root}: {WEIGHTS_NAME} changed while stamping")
    again = _tensors(after)
    if again.keys() != tensors.keys() or any(again[k] != tensors[k] for k in tensors):
        raise AssertionError(f"{root}: a tensor changed while stamping")
    return Stamped(root, before, artifact_digest(root), digest)


def _tensors(blob: bytes) -> dict[str, bytes]:
    """Every tensor's raw bytes, by name: equal bytes are the strictest equality there is."""
    from safetensors.numpy import load

    return {name: array.tobytes() for name, array in load(blob).items()}


def main(argv: list[str] | None = None) -> int:
    folders = sys.argv[1:] if argv is None else argv
    if not folders:
        print("usage: python -m royaleimitate.stamp FOLDER [FOLDER ...]")
        return 2
    for folder in folders:
        done = stamp_actor_digest(folder)
        state = "unchanged" if done.old_sha256 == done.new_sha256 else "stamped"
        print(
            f"{done.folder}: {state}; sha256 {done.old_sha256} -> {done.new_sha256}; "
            f"actor_digest {done.actor_digest}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
