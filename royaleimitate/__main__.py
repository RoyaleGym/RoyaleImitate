"""``python -m royaleimitate``: the same command as the ``royaleimitate`` script, for a shell where
the virtual environment's scripts folder is not on PATH."""

from .cli import main

raise SystemExit(main())
