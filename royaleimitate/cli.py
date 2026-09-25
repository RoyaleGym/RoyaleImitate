"""``royaleimitate``: the commands that make what the imitation sections read.

- ``artifact-digest <folder>`` prints the digest a config names an artifact folder by.
- ``fit-field-reference`` fits the play/wait model a ``field_mlp`` reference loads.

A training run itself is still ``royalelearn train``: RoyaleLearn finds this package through the
``warm_start`` and ``imitation`` sections of the run's config.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="royaleimitate",
        description="Make the artifacts RoyaleLearn's imitation sections read.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    digest = commands.add_parser(
        "artifact-digest", help="print the digest a config names an artifact folder by"
    )
    digest.add_argument("folder", type=Path)
    digest.set_defaults(handler=_artifact_digest)

    fit = commands.add_parser(
        "fit-field-reference",
        help="fit the play/wait model a field_mlp reference loads",
    )
    fit.add_argument("--rows", type=Path, required=True, help="an .npz of field columns")
    fit.add_argument(
        "--fields", required=True, help="comma-separated observation vector field names"
    )
    fit.add_argument("--out", type=Path, required=True, help="a new folder for the artifact")
    fit.add_argument("--hidden", default="32,32", help="hidden layer widths, comma-separated")
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--seed", type=int, default=20260924)
    fit.set_defaults(handler=_fit_field_reference)
    return parser


def _fit_field_reference(args: argparse.Namespace) -> int:
    """Fit the play/wait model and write it as a field-model artifact."""
    from .fit import FitConfig, fit_field_reference

    fields = [name.strip() for name in args.fields.split(",") if name.strip()]
    hidden = [int(width) for width in args.hidden.split(",") if width.strip()]
    fit_field_reference(
        args.rows,
        fields,
        args.out,
        config=FitConfig(hidden=hidden, epochs=args.epochs, seed=args.seed),
    )
    return 0


def _artifact_digest(args: argparse.Namespace) -> int:
    """The folder digest: over every file in the folder, by relative path."""
    from .artifacts import artifact_digest

    print(artifact_digest(args.folder))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from royalelearn.extensions import PreflightError

    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except PreflightError as exc:
        print(f"royaleimitate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
