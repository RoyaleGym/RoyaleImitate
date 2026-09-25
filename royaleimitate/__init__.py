"""Imitation learning for RoyaleLearn, as an optional add-on.

Two config sections, found by RoyaleLearn through this package's entry points:

- ``warm_start``: start the actor from saved weights, check them against their own probe rows,
  and hold the actor still at first while the critic catches up;
- ``imitation``: anchor the policy to reference policies with the reference-KL regulariser and
  its adaptive coefficient.

And the tools that make what they read: saved-policy folders (``artifacts``), field models
(``field_model``, ``fit``), and demonstration shards (``shards``, ``split``). And ``public_log``:
the observation's fair fields from a timed log of card plays, with no engine running.
"""

from __future__ import annotations

__version__ = "0.1.0"
