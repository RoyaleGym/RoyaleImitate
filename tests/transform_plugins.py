"""A policy transform for ``tests/test_policy_transform.py``.

Named in a config as ``plugin:transform_plugins:sharpen``.
"""

from __future__ import annotations

from typing import Any


def sharpen(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    """A temperature of one half."""
    return logits * 2.0
