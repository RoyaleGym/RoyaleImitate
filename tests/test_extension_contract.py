"""What RoyaleLearn requires of this package as an extension, checked the way RoyaleLearn checks it.

The entry points are the ones pyproject.toml declares, found through the install metadata the
session's conftest writes for this checkout; the providers are resolved by RoyaleLearn's own
``providers_for``.
"""

from __future__ import annotations

import msgspec
import pytest

import royaleimitate
from conftest import declared_entry_points
from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.extensions import EXTENSION_API_VERSION, active_extensions, providers_for


def test_the_declared_sections_are_the_two_this_package_provides() -> None:
    assert declared_entry_points() == {
        "imitation": "royaleimitate.extension:EXTENSION",
        "warm_start": "royaleimitate.warm_start:EXTENSION",
    }


def test_each_section_resolves_to_an_extension_royalelearn_accepts() -> None:
    """providers_for refuses another API version, a name that is not its key, and a section
    type that would accept a misspelt field; it accepts both of these."""
    providers = providers_for(["imitation", "warm_start"])
    for name, provider in providers.items():
        extension = provider.extension
        assert extension.name == name
        assert extension.api_version == EXTENSION_API_VERSION
        assert extension.package is royaleimitate
        assert not provider.builtin
        assert provider.distribution == "royaleimitate"
        assert issubclass(extension.section_type, msgspec.Struct)
        assert extension.section_type.__struct_config__.forbid_unknown_fields


def test_a_config_naming_the_sections_loads_them_from_this_package() -> None:
    loaded = cfg.load_config(
        {"warm_start": {"actor_lr_scale": {"kind": "constant", "value": 1.0}}, "imitation": {}}
    )
    assert sorted(a.name for a in active_extensions(loaded)) == ["imitation", "warm_start"]


def test_a_misspelt_field_inside_a_section_is_refused() -> None:
    with pytest.raises(msgspec.ValidationError, match="inits"):
        cfg.load_config({"warm_start": {"inits": {}}})


def test_a_config_an_older_build_wrote_for_an_il_run_loads() -> None:
    """Every IL config.json of 7e93217..S2 wrote ``init`` and ``actor_lr_scale`` inside the
    imitation block, null when unset; RoyaleLearn's one-release shim drops the nulls."""
    said: list[str] = []
    loaded = cfg.load_config(
        {"imitation": {"init": None, "actor_lr_scale": None, "references": {}, "regularisers": []}},
        say=said.append,
    )
    assert [a.name for a in active_extensions(loaded)] == ["imitation"]
    assert said and "imitation.init" in said[0] and "imitation.actor_lr_scale" in said[0]


def test_a_non_null_init_in_the_old_place_is_refused_with_its_new_home() -> None:
    with pytest.raises(PreflightError, match=r"imitation\.init is now warm_start\.init"):
        cfg.load_config({"imitation": {"init": {"path": "x", "sha256": "y"}}})
