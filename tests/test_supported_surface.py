"""This package reaches into RoyaleLearn through its supported surface and nowhere else, and it
does not reach into the private dataset readers at all.

``royalelearn.extensions.__all__`` is what RoyaleLearn promises to keep; anything else in it can
move in any release. And ``datasets/`` in this folder is a separate private repository for
readers of data that declares no licence: this package must work, and pass these tests, on a
clone that does not have it.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import royalelearn.extensions as surface

PACKAGE = Path(__file__).resolve().parents[1] / "royaleimitate"
ROOT = PACKAGE.parent


def _imports(path: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend((alias.name, "") for alias in node.names)
    return found


def test_every_royalelearn_import_is_on_the_supported_surface() -> None:
    outside = sorted(
        f"{path.name}: {module}.{name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for module, name in _imports(path)
        if module.split(".")[0] == "royalelearn"
        and (module != "royalelearn.extensions" or name not in surface.__all__)
    )
    assert outside == [], "imports RoyaleLearn does not promise to keep"


def test_nothing_imports_the_private_dataset_readers() -> None:
    reached = sorted(
        f"{path.name}: {module}"
        for path in sorted(PACKAGE.rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
        for module, _ in _imports(path)
        if module.split(".")[0] == "royaleimitate_data"
    )
    assert reached == []


def test_the_dataset_folder_is_not_part_of_this_repository() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "datasets"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert tracked == ""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "datasets/"], cwd=ROOT, check=False
    ).returncode
    assert ignored == 0, "datasets/ must stay ignored so no commit can take it in"
