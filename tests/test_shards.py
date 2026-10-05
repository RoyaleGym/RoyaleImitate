"""Section 19.10: demonstration shards.

The round trip is the property: rows written from the environment's own observations, read back
and packed by the reading run's codec, are byte for byte the rows the codec makes from those
observations directly. Everything else here is a refusal a shard owes its reader -- another
environment, another engine without a reason, a part that changed, a write that never finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royaleimitate.shards import (
    FLAG_PROJECTED,
    MANIFEST_NAME,
    ShardContext,
    ShardReader,
    ShardWriter,
)
from royalelearn.errors import PreflightError

torch = pytest.importorskip("torch")

from royalelearn.testing import coordinator, tiny_config  # noqa: E402


@pytest.fixture(scope="module")
def run_side(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A run's context and codec, and forty of its environment's own observations."""
    root = tmp_path_factory.mktemp("shards")
    config = tiny_config(root / "run")
    context = ShardContext.of_config(config)
    vec = config.env.build_vec(1)
    observations: list[dict[str, np.ndarray]] = []
    try:
        batch = vec.reset(seed=11)[0]
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs = {key: np.array(value[0]) for key, value in batch.items()}
            observations.append(obs)
            legal = np.flatnonzero(obs["action_mask"])
            action = np.array([int(rng.choice(legal))] * vec.num_envs)
            batch = vec.step(action)[0]
    finally:
        vec.close()
    with coordinator(config) as run:
        codec = run.row_codec()
        yield context, codec, observations


def _write(root: Path, context: ShardContext, observations: list[dict], **kwargs: Any) -> Path:
    with ShardWriter(root, context, rows_per_part=kwargs.pop("rows_per_part", 16)) as writer:
        for index, obs in enumerate(observations):
            legal = np.flatnonzero(obs["action_mask"])
            writer.add(
                obs,
                action=int(legal[index % legal.size]),
                weight=1.0,
                group=index // 4,
                seat=index % 2,
                tick=90 + 10 * index,
                flags=FLAG_PROJECTED if index % 5 == 0 else 0,
                reward=0.01 * index,
            )
    return root / context.engine_key


def test_rows_come_back_exactly_and_pack_as_the_run_packs_them(
    run_side: Any, tmp_path: Path
) -> None:
    """Plant: store one quantum high and the packed rows stop matching."""
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations)
    reader = ShardReader(directory, context)
    assert reader.manifest.rows == len(observations)
    assert [part.rows for part in reader.manifest.parts] == [16, 16, 8]
    assert reader.manifest.flag_counts["projected"] == 8
    back: list[np.ndarray] = []
    for index in range(len(reader.manifest.parts)):
        rows, scalars = reader.packed(index, codec)
        back.append(rows)
        assert scalars["action"].dtype == np.int16
    direct = codec.pack(observations)
    assert np.array_equal(np.concatenate(back), direct)
    first = reader.observations(reader.part(0))[0]
    assert np.array_equal(first["spatial"], observations[0]["spatial"])
    assert np.array_equal(
        first["action_mask"].astype(bool), observations[0]["action_mask"].astype(bool)
    )


def test_a_value_the_fixed_point_cannot_hold_is_refused_by_plane(
    run_side: Any, tmp_path: Path
) -> None:
    context, _codec, observations = run_side
    obs = {key: value.copy() for key, value in observations[0].items()}
    obs["spatial"][0, 0, 0] = 0.0004
    with (
        pytest.raises(PreflightError, match=r"spatial plane 0 \("),
        ShardWriter(tmp_path, context) as writer,
    ):
        writer.add(obs, action=0, weight=1.0, group=0, seat=0, tick=90)


def test_a_write_that_did_not_finish_is_refused(run_side: Any, tmp_path: Path) -> None:
    context, _codec, observations = run_side
    writer = ShardWriter(tmp_path, context, rows_per_part=4)
    for obs in observations[:8]:
        writer.add(obs, action=0, weight=1.0, group=0, seat=0, tick=90)
    with pytest.raises(PreflightError, match=f"no {MANIFEST_NAME}"):
        ShardReader(tmp_path / context.engine_key, context)
    with pytest.raises(FileExistsError):
        ShardWriter(tmp_path, context)


def test_another_environment_is_refused_by_name(run_side: Any, tmp_path: Path) -> None:
    context, _codec, observations = run_side
    directory = _write(tmp_path, context, observations[:4])
    other = msgspec.structs.replace(context, obs_digest="0" * 16, catalogue_sha256="1" * 64)
    with pytest.raises(PreflightError) as refused:
        ShardReader(directory, other)
    assert "obs_digest" in str(refused.value) and "catalogue_sha256" in str(refused.value)


def test_another_engine_needs_a_reason_and_the_reason_is_kept(
    run_side: Any, tmp_path: Path
) -> None:
    context, _codec, observations = run_side
    directory = _write(tmp_path, context, observations[:4])
    rebuilt = msgspec.structs.replace(context, binary_sha256="f" * 64)
    assert rebuilt.engine_key != context.engine_key
    with pytest.raises(PreflightError, match="binary_sha256"):
        ShardReader(directory, rebuilt)
    reader = ShardReader(directory, rebuilt, allow_engine_mismatch="pilot on the older build")
    assert reader.provenance["engine_mismatch"] == "pilot on the older build"
    assert reader.provenance["engine_key"] == context.engine_key


def test_a_part_that_changed_after_writing_is_refused(run_side: Any, tmp_path: Path) -> None:
    context, codec, observations = run_side
    directory = _write(tmp_path, context, observations)
    part = directory / "part-00001.npz"
    blob = bytearray(part.read_bytes())
    blob[-1] ^= 0xFF
    part.write_bytes(bytes(blob))
    reader = ShardReader(directory, context)
    reader.part(0)
    with pytest.raises(PreflightError, match="changed after the shard was written"):
        reader.packed(1, codec)


def test_a_row_from_another_arena_is_refused_on_read(run_side: Any, tmp_path: Path) -> None:
    """The codec supplies the run's static planes on decode, so a row's own must match them."""
    context, codec, observations = run_side
    if not context.static_planes:
        pytest.skip("this observation declares no static plane")
    moved = []
    for obs in observations[:4]:
        copy = {key: value.copy() for key, value in obs.items()}
        copy["spatial"][context.static_planes[0]] = 1.0
        moved.append(copy)
    directory = _write(tmp_path, context, moved)
    with pytest.raises(PreflightError, match="static planes differ"):
        ShardReader(directory, context).packed(0, codec)


def test_batches_cover_each_split_once_in_a_seeded_order(run_side: Any, tmp_path: Path) -> None:
    context, codec, observations = run_side
    many = observations * 5
    directory = _write(tmp_path, context, many, rows_per_part=32)
    reader = ShardReader(directory, context)

    def ticks(split: str, epoch: int) -> list[int]:
        seen: list[int] = []
        for batch in reader.batches(
            codec, split=split, batch_rows=16, seed=3, epoch=epoch, window_rows=64
        ):
            assert batch.obs.mask.shape[0] == batch.actions.shape[0]
            seen.extend(int(g) for g in batch.groups)
        return seen

    train, held = ticks("train", 0), ticks("validation", 0)
    everything = sorted(train + held)
    assert everything == sorted(index // 4 for index in range(len(many)))
    assert not set(train) & set(held), "a group was split across train and validation"
    from royaleimitate.split import is_validation

    groups = np.array(sorted(set(everything)), dtype=np.uint64)
    assert set(held) == set(groups[is_validation(groups)].tolist())
    assert ticks("train", 0) == train
    assert ticks("train", 1) != train
    assert sorted(ticks("train", 1)) == sorted(train)


def test_a_row_the_writer_cannot_vouch_for_is_refused(run_side: Any, tmp_path: Path) -> None:
    context, _codec, observations = run_side
    obs = observations[0]
    illegal = int(np.flatnonzero(~obs["action_mask"].astype(bool))[0])
    with ShardWriter(tmp_path, context) as writer:
        with pytest.raises(ValueError, match="not legal"):
            writer.add(obs, action=illegal, weight=1.0, group=0, seat=0, tick=90)
        with pytest.raises(ValueError, match="not named"):
            writer.add(obs, action=0, weight=1.0, group=0, seat=0, tick=90, flags=1 << 9)
    with pytest.raises(PreflightError, match="from 1 << 8 up"):
        ShardWriter(tmp_path / "other", context, flag_names={4: "tier_o"})


def test_spell_ids_are_stored_beside_the_card_ids_and_pack_as_the_run_packs_them(
    tmp_path: Path,
) -> None:
    """With ``spell_identity`` on, its planes are a column of their own, read back exactly, and
    the reading run's codec packs them into its id region. Plant: drop the column and the packed
    rows stop matching."""
    from royalelearn.rollout.envspec import ComponentSpec

    config = tiny_config(tmp_path / "run")
    builder = config.env.obs_builder
    switches = {"card_identity": True, "spell_identity": True, "spell_aim_after_ticks": 20}
    env = msgspec.structs.replace(
        config.env, obs_builder=ComponentSpec(builder.cls, {**builder.kwargs, **switches})
    )
    config = msgspec.structs.replace(config, env=env)
    context = ShardContext.of_config(config)
    assert context.spell_id_planes == 4 and context.card_id_planes == 2
    vec = config.env.build_vec(1)
    observations: list[dict[str, np.ndarray]] = []
    try:
        batch = vec.reset(seed=3)[0]
        for index in range(20):
            obs = {key: np.array(value[0]) for key, value in batch.items()}
            # Spells in flight are rare on a short mock battle, so every id is put on a plane.
            obs["spell_ids"] = np.full_like(obs["spell_ids"], index % 7)
            observations.append(obs)
            batch = vec.step(np.zeros(vec.num_envs, dtype=np.int64))[0]
    finally:
        vec.close()
    directory = _write(tmp_path / "shards", context, observations)
    reader = ShardReader(directory, context)
    back = reader.observations(reader.part(0))
    assert all(
        np.array_equal(row["spell_ids"], obs["spell_ids"])
        for row, obs in zip(back, observations, strict=False)
    )
    with coordinator(config) as run:
        codec = run.row_codec()
        packed = np.concatenate(
            [reader.packed(i, codec)[0] for i in range(len(reader.manifest.parts))]
        )
        assert np.array_equal(packed, codec.pack(observations))


class UnitTypedMock:
    """Built lazily: a MockEngine that says each entity's own unit type (royalesim 0.1.17 on)."""

    UNITS = ("Archer", "Giant", "Goblin", "KingTower", "Knight", "PrincessTower")

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from royalegym.mock_engine import MockEngine
        from royalegym.protocol import EntityKind

        units = cls.UNITS

        class _Typed(MockEngine):
            def unit_types(self) -> list[str]:
                return list(units)

            def state(self) -> Any:
                s = super().state()

                def typed(e: Any) -> Any:
                    if e.kind == EntityKind.KING_TOWER:
                        name = "KingTower"
                    elif e.kind == EntityKind.PRINCESS_TOWER:
                        name = "PrincessTower"
                    else:
                        name = units[e.card_id % 3]
                    return msgspec.structs.replace(e, unit_type=units.index(name))

                return msgspec.structs.replace(s, entities=[typed(e) for e in s.entities])

        return _Typed(*args, **kwargs)


def test_unit_ids_are_stored_as_their_own_column_and_pack_as_the_run_packs_them(
    tmp_path: Path,
) -> None:
    """Plant: drop the column on read and the packed rows stop matching."""
    from royalelearn.rollout.envspec import ComponentSpec

    config = tiny_config(tmp_path / "run")
    builder = config.env.obs_builder
    switches = {"card_identity": True, "unit_identity": True}
    env = msgspec.structs.replace(
        config.env,
        engine=ComponentSpec("test_shards.UnitTypedMock", {}),
        obs_builder=ComponentSpec(builder.cls, {**builder.kwargs, **switches}),
    )
    config = msgspec.structs.replace(config, env=env, extra_component_modules=["test_shards."])
    context = ShardContext.of_config(config)
    assert context.unit_id_planes == 2
    vec = config.env.build_vec(1, ("test_shards.",))
    observations: list[dict[str, np.ndarray]] = []
    try:
        batch = vec.reset(seed=3)[0]
        for index in range(20):
            obs = {key: np.array(value[0]) for key, value in batch.items()}
            obs["unit_ids"] = np.full_like(obs["unit_ids"], index % 6)
            observations.append(obs)
            batch = vec.step(np.zeros(vec.num_envs, dtype=np.int64))[0]
    finally:
        vec.close()
    directory = _write(tmp_path / "shards", context, observations)
    reader = ShardReader(directory, context)
    back = reader.observations(reader.part(0))
    assert all(
        np.array_equal(row["unit_ids"], obs["unit_ids"])
        for row, obs in zip(back, observations, strict=False)
    )
    with coordinator(config) as run:
        codec = run.row_codec()
        packed = np.concatenate(
            [reader.packed(i, codec)[0] for i in range(len(reader.manifest.parts))]
        )
        assert np.array_equal(packed, codec.pack(observations))
