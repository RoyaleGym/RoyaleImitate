"""``ShardReader``'s cache of packed parts: the second epoch reads no part file again.

``packed`` turns one part into the reading run's packed rows: it reads and inflates the part,
rebuilds every observation and packs it, row by row. On a processor that is most of an epoch,
and the result is the same every epoch. So it is kept: in memory while the dataset's packed
rows fit ``cache_memory_mb``, else as uncompressed ``.npy`` files beside the shard, which the
next run on the same rows and codec reads too, else not at all. ``batches`` is untouched, so the
order of the rows and of the batches is what it was.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from royaleimitate.shards import ShardReader

torch = pytest.importorskip("torch")

from test_shards import _write, run_side  # noqa: E402,F401


def _epochs(reader: ShardReader, codec: Any, epochs: int = 2) -> list[list[Any]]:
    """Every batch of ``epochs`` epochs of the train split, as plain arrays."""
    out = []
    for epoch in range(epochs):
        batches = []
        for batch in reader.batches(
            codec, split="train", batch_rows=16, seed=3, epoch=epoch, window_rows=64
        ):
            batches.append(
                (
                    batch.obs.spatial.cpu().numpy(),
                    batch.obs.vector.cpu().numpy(),
                    batch.obs.mask.cpu().numpy(),
                    batch.actions.cpu().numpy(),
                    batch.weights.cpu().numpy(),
                    batch.groups,
                )
            )
        out.append(batches)
    return out


def _same(a: list[list[Any]], b: list[list[Any]]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if len(x) != len(y):
            return False
        for u, v in zip(x, y, strict=True):
            if not all(np.array_equal(p, q) for p, q in zip(u, v, strict=True)):
                return False
    return True


def _counting(reader: ShardReader, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    reads: list[int] = []
    real = reader.part

    def part(index: int) -> dict[str, np.ndarray]:
        reads.append(index)
        return real(index)

    monkeypatch.setattr(reader, "part", part)
    return reads


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_a_cached_reader_gives_the_same_batches_and_reads_each_part_once(
    run_side: Any,  # noqa: F811 - the shard tests' fixture, imported above
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Plant: no cache, and the second epoch reads every part again."""
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations * 5, rows_per_part=32)
    plain = _epochs(ShardReader(directory, context, cache=None), codec)
    reader = ShardReader(directory, context, cache=mode)
    reads = _counting(reader, monkeypatch)
    cached = _epochs(reader, codec)
    assert _same(plain, cached)
    assert sorted(reads) == list(range(len(reader.manifest.parts)))
    assert reader.cache_mode == mode


def test_the_disk_cache_is_read_by_the_next_reader_and_keyed_by_the_part(
    run_side: Any,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run on the same rows and codec reads no part; a part rewritten since is read."""
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations * 5, rows_per_part=32)
    first = _epochs(ShardReader(directory, context, cache="disk"), codec, epochs=1)
    again = ShardReader(directory, context, cache="disk")
    reads = _counting(again, monkeypatch)
    assert _same(first, _epochs(again, codec, epochs=1))
    assert reads == []
    # Rows rewritten under the same names: the manifest's sha256s move, and so do the keys.
    other = tmp_path / "other"
    _write(other, context, list(reversed(observations * 5)), rows_per_part=32)
    for path in (other / context.engine_key).iterdir():
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    fresh = ShardReader(directory, context, cache="disk")
    reads = _counting(fresh, monkeypatch)
    _epochs(fresh, codec, epochs=1)
    assert sorted(reads) == list(range(len(fresh.manifest.parts)))


def test_a_cached_part_cannot_be_changed_by_a_caller(
    run_side: Any,  # noqa: F811
    tmp_path: Path,
) -> None:
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations * 2, rows_per_part=32)
    reader = ShardReader(directory, context, cache="memory")
    rows, scalars = reader.packed(0, codec, split="all")
    with pytest.raises(ValueError):
        rows[0, 0] = 1
    with pytest.raises(ValueError):
        scalars["action"][0] = 1


def test_auto_takes_memory_then_disk_then_nothing(
    run_side: Any,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """By the dataset's packed size against the memory budget, then against the free disk."""
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations * 2, rows_per_part=32)
    reader = ShardReader(directory, context, cache="auto")
    assert reader.plan_cache(codec).mode == "memory"
    small = ShardReader(directory, context, cache="auto", cache_memory_mb=0)
    plan = small.plan_cache(codec)
    assert plan.mode == "disk" and plan.where is not None
    import royaleimitate.shards as shards

    full = SimpleNamespace(total=10**9, used=10**9, free=0)
    monkeypatch.setattr(shards.shutil, "disk_usage", lambda _path: full)
    none = ShardReader(directory, context, cache="auto", cache_memory_mb=0)
    assert none.plan_cache(codec).mode is None
    assert "free" in none.plan_cache(codec).said


def test_false_turns_the_cache_off_and_the_disk_line_says_how(
    run_side: Any,  # noqa: F811
    tmp_path: Path,
) -> None:
    """The disk cache writes beside a user's own rows, so turning it off is one word, and the
    line that announces it says which, and how big it is."""
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations * 2, rows_per_part=32)
    off = ShardReader(directory, context, cache=False)
    assert off.plan_cache(codec).mode is None
    disk = ShardReader(directory, context, cache="disk").plan_cache(codec)
    assert "MB" in disk.said and "cache=False" in disk.said
    with pytest.raises(ValueError, match="cache"):
        ShardReader(directory, context, cache="sometimes")
