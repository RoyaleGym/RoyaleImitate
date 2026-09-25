# RoyaleImitate

Imitation learning for the Royale stack, as an optional add-on to RoyaleLearn. Install it next to
RoyaleLearn when you want a policy to start from human play; leave it out and RoyaleLearn behaves as
if it did not exist.

**Status (2026-09-24): being assembled.** The generic package (`royaleimitate/`: reference policies
and the reference-KL anchor, warm-start from a saved policy, demonstration shards, the replay driver,
behaviour cloning) moves here from RoyaleLearn by the steps in the extraction plan. Until that lands,
this repository holds only this README.

A run uses it by giving its config an `imitation` section; RoyaleLearn finds this package through
that section and refuses at start, naming the section, if the package is not installed.

`datasets/` is not part of this repository. On machines that have it, it is a separate private
repository of dataset-specific readers, kept apart because the datasets they read carry no licence.
