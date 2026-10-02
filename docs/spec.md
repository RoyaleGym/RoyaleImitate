# RoyaleImitate: the contract

This is section 19 of RoyaleLearn's `docs/harness-spec.md`, "Learning from demonstrations", as it
stood when the code moved here, under the same numbers. The parts that stay RoyaleLearn's are in
that document: how an extension section is found and recorded (19.1), the freeze a section can
schedule (19.5), and export (19.13). Section numbers elsewhere in both repositories refer to this
one numbering.

## 19. Learning from demonstrations

A run can start from a policy cloned from demonstrations, and can be held near a reference policy
while it learns. This section is the contract for that machinery. It is generic: a demonstration is
any recorded play stored as shard rows (19.10).

It lands as six packages, named here so that a config, a commit and a test can refer to them.
Not all of it is built (checked 2026-09-28 against this repo's code):

| Package | What | Section | Built |
| --- | --- | --- | --- |
| L1 | the `warm_start` and `imitation` config sections, their identity, actor init and the freeze | 19.1-19.5 | yes |
| L2 | reference policies, the reference-KL regulariser and its adaptive coefficient | 19.6-19.9 | yes |
| L4 | demonstration shards: the stored rows, keyed to the engine | 19.10 | yes, `royaleimitate.shards` |
| L5 | recording and cloning, the field-reference fit and the `demo_bc` regulariser | 19.12 | the fit, as `royaleimitate fit-field-reference` |
| L6 | checkpoint export and `royalelearn evaluate` | 19.13 | no |

The commands that exist are `royaleimitate artifact-digest` and `royaleimitate
fit-field-reference`. The sections below still describe the rest, as specified.

One more part sits outside the six, and outside RoyaleLearn's section 19:
`royaleimitate.public_log.PublicLogMemory` rebuilds the observation's fair fields from a timed log
of card plays, with no engine running. It is described in [public-log.md](public-log.md).

When the sections are absent nothing in this section runs, and a run's config.json, identity and
`run_id` are exactly what they were before they existed. That is tested.

### 19.1 The `warm_start` and `imitation` sections

Two optional top-level config sections, each owned by an extension (`royalelearn.extensions`): a key
RoyaleLearn does not own is looked up among the installed extensions, and one nothing provides is
refused by name. Top level rather than inside `ppo`, because most of it is not about the PPO update
(an init, a set of reference files), and because `algo_digest` hashes the `ppo` block whole. Each
section carries its own alarm thresholds, which stay out of the identity as the core's do.

```json
{
"warm_start": {
  "init": {"path": "artifacts/my-policy", "sha256": "<artifact digest>"}
},
"imitation": {
  "references": {
    "teacher": {"kind": "snapshot", "path": "artifacts/my-policy", "sha256": "<artifact digest>"}
  },
  "regularisers": [
    {"kind": "reference_kl", "name": "teacher", "reference": "teacher"}
  ]
}
}
```

Every other field has a default: a regulariser's `budget` (a constant 0.1), its `coef` (starting at
1.0) and its `factor` (`joint`), and each section's alarm thresholds.

- `init` (optional): the actor's starting weights (19.4).
- `actor_lr_scale` (optional, a schedule): multiplies the actor's learning rate (19.5). Absent is 1.
- `references`: named reference policies (19.6). A regulariser names one.
- `regularisers`: `reference_kl` (19.7), and `demo_bc` (19.12), which is specified and not built:
  a config that names it is refused. Each has a `name` that is unique
  in the list and becomes the metric segment `imitation/<name>/...`.

Every schedule is an ordinary `ScheduleSpec` in env steps, the harness's one clock, so a resume
restores every value from `cumulative_env_steps` and nothing else.

`check_consistency` refuses, by name: a regulariser naming a reference that is not declared; a
`field_mlp` reference under `factor: joint` (it has no opinion about cards or tiles); two
regularisers with one name; a name that is not one metric segment; `max` below the smallest value
`min` takes; a negative coefficient, budget or scale; `up` or `down` or `band` at or below one; and
an `exclude_when` field the run's vector layout does not declare (checked at preflight, where the
layout is known).

### 19.2 Artifacts, and what `sha256` means

An artifact is a folder. An actor artifact (a cloned policy, an exported checkpoint) is the snapshot
layout of section 11.6, `actor.safetensors` and `spec.json`, plus `probe.safetensors` when it is
meant to be loaded as an init. A field-MLP artifact is `model.safetensors` and `spec.json`.

Unlike a pool snapshot, an actor artifact's weights are float32. A pool snapshot is played against
and never trained from, so half precision is enough; an init is trained from, and the init-identity
control (19.15) needs the file to hold the seeded weights exactly. `SnapshotSpec.meta["dtype"]`
says which it is, and the snapshot store loads either.

`sha256` is the artifact digest: the sha256 of the canonical JSON of `{relative path: sha256 of the
file's bytes}` over every file in the folder. Over the folder rather than over the weights, because
the other files change numbers too: a field-MLP's normalisation lives in `spec.json` and an init's
self-test in `probe.safetensors`. `royaleimitate artifact-digest <folder>` prints it, and every tool
that writes an artifact prints it as it finishes.

A shard directory (19.10) is the exception, because it can be many gigabytes: its digest is the
sha256 of its `manifest.json`, which lists every part file's own sha256, and the reader checks each
part against the manifest when it opens it.

### 19.3 Identity

`RunIdentity` has `extensions: dict[str, ExtensionRecord] | None = None`, one record per section the
run uses, and the struct is encoded with `omit_defaults`, so a run without a section encodes exactly
as before and keeps its `run_id`. Every other field is always set when an identity is computed, so
`omit_defaults` removes nothing else.

A record's `digest` is of the section with every `path` and its `alarms` removed. The section carries
each file's digest, so the content of every referenced file is in the identity, and moving a folder
is not a new experiment. A section provided by an installed package also records the package's
distribution, `__version__` and commit; a package whose commit cannot be named refuses the run.

At every start, fresh or resumed, each referenced folder is hashed and compared with the digest the
config states, and a difference is refused with the path, the stated digest and the digest found.
Without that check the identity would be only as true as the config's claim about the file.

`RunIdentity` is a fixed field list, so a new config section is invisible to it unless a field is
added. This is that field.

### 19.4 Initialising the actor

On a **fresh** start, in this order:

1. The model is built from `master_seed` exactly as without the block, so the critic's initial
   weights equal the from-scratch run's for the same seed.
2. The artifact's digest is checked (19.3).
3. Its `spec.json` is checked against the run's with `check_compatible`, plus `action_digest`.
4. The weights are loaded into the actor with `strict=True`.
5. **The probe-logit self-test.** The artifact's probe rows, stored as the run's codec packed them,
   are decoded through the run's codec and put through the run's freshly loaded actor. The masked
   log-probabilities on every legal action must equal the ones recorded by the tool that wrote the
   artifact to within `self_test_atol`. A failure names the worst row and the largest difference.
   An init artifact without probe rows is refused.

On **resume** the checkpoint's weights win, and only the digest is checked.

The self-test is what makes steps 3 and 4 mean something. A compatible spec says the tensors have
the right shapes and the observation the right layout; only a forward on known rows says the network
computes the function that was saved.

### 19.5 Freezing and ramping the actor

RoyaleLearn's `docs/harness-spec.md` section 19.5: the freeze is RoyaleLearn's mechanism, and
the `warm_start` section supplies its schedule and its two alarms' thresholds.

### 19.6 Reference policies

A reference is evaluated on the rows the actor is trained on, under `no_grad`, with the run's
autocast, and never trained.

- **`snapshot`**: an actor artifact (19.2), loaded into a second copy of the run's actor and checked
  as an init is (steps 2-4 of 19.4), plus the self-test at 1e-5 when the artifact carries probe
  rows. Its distribution is a `MaskedCategorical` over the row's own mask, so it is a full policy
  over the legal set.
- **`field_mlp`**: a small MLP over named fields of the observation vector, giving the logit of
  p(play) on a row. Its `spec.json` lists each field's name and width, the normalisation, the
  layer widths and activation. At load every field must exist in the run's vector layout with that
  width, or it is refused naming the field. It defines only the play/wait marginal:
  p_ref(no-op) = sigmoid(-z).

A reference is a function of the observation alone, so it adds no state to a checkpoint.

### 19.7 The reference-KL regulariser

For each regulariser, on each choice row that survives `exclude_when`:

- `factor: joint`: KL(π_ref ‖ π_θ) = Σ over legal a of p_ref(a) (log p_ref(a) − log π_θ(a)),
  exact over the masked legal set.
- `factor: noop_marginal`: the KL between the two play/wait Bernoullis, in log space:
  log(1 − p) is `log(-expm1(log p))`, as `hold_gap` computes it.

**Forward**, KL(π_ref ‖ π_θ): it charges π_θ for removing probability where the reference puts it,
which covers a collapse onto one tile and a card the policy stops playing. The reverse direction
charges a collapse onto one of the reference's own modes almost nothing.

**Where it is added.** In both actor paths of the update (`_minibatch` and
`_critic_then_the_rows_that_can_move`), as λ × Σ_rows KL × `actor_scale`, the same scale the policy
term uses. So its weight relative to the policy term is the same under every value of
`ppo.forced_rows`, and minibatch size stays a pure memory knob; both are tested. A row with one
legal action has zero KL under either factor, because both distributions are the same point mass,
so choice rows lose nothing by being the only rows.

**`exclude_when`**: a list of conditions `{"field", "index", "op", "value"}` on elements of the
observation vector, `op` one of `<`, `<=`, `>`, `>=`. A row is left out when all of them hold. It
is how a regulariser is kept off a part of the game its reference was not fitted on.

**Decomposed for reporting**, exactly, by the chain rule over the action layout (no-op, then hand
slot, then tile): `kl = kl_noop + kl_card + kl_tile`, where `kl_card` is p_ref(play) times the KL
of the slot given play, and `kl_tile` is the reference-weighted KL of the tile given the slot. The
identity is tested to float tolerance.

**`imitation/<name>/grad_ratio`**: ‖∇ of the KL sum times `actor_scale`‖ / ‖∇ of the policy term‖
over the actor's parameters, before the coefficient (lambda), on the first minibatch of each
iteration that has a choice row. It is for
setting the coefficient's start. At π_θ = π_ref the forward KL's gradient is exactly zero, so the
ratio is only informative some iterations after the two have parted; a dry run reads it there.

### 19.8 The adaptive coefficient

Once per iteration, from what that iteration's update measured:

```text
t      = the iteration's schedule clock (cumulative env steps, as every other schedule)
lam_t  = clamp(lam, min(t), max)            # the coefficient this iteration's loss uses
m      = mean KL over the choice rows the regulariser covered in EPOCH 1
kappa  = budget(t)
lam    = lam_t * up        if m > band * kappa
         lam_t / down      if m < kappa / band
         lam_t             otherwise
lam    = clamp(lam, min(t), max)            # stored in the checkpoint
```

- `m` is measured in the first epoch because that is the policy the iteration's rollouts came from,
  with the smallest possible drift inside it, and it costs nothing: the forward is already run.
- When the actor is frozen, or no row was covered, there is no `m` and λ does not move.
- λ is update state like `BackoffState` and is saved in the update's checkpoint folder. A resume
  restores it; a checkpoint written without the block restores the configured `start`.
- `min`, `max` and `budget` are schedules; λ stays inside `[min(t), max]`.

`imitation/<name>/lambda` is the value the iteration used and `imitation/<name>/budget` is κ.

### 19.9 Metrics and alarms

Keys, each present only when its regulariser is configured:

- `imitation/<name>/{kl, kl_noop, kl_card, kl_tile}`: epoch-1 means over the covered choice rows;
  `kl_card` and `kl_tile` are absent under `noop_marginal`.
- `imitation/<name>/{lambda, lambda_at_max, budget, grad_ratio, rows_frac}`,
  `imitation/<name>/top1_agree` (the share of covered rows where the two argmaxes agree; under
  `noop_marginal`, whether both say play), and `imitation/<name>/ref_p_noop`.
- `ppo/actor_lr_scale` and `ppo/actor_frozen` when `actor_lr_scale` is set;
  `ppo/ev_at_unfreeze` on the first row after a frozen stretch, and
  `ppo/iterations_since_unfreeze` on every unfrozen row after one.
- `env/play_rate_by_elixir/{k}`, whether or not the block is present: the share of the learner's
  sampled choice rows at elixir floor `k` (0-10) where it played. Computed where
  `env/mean_elixir_at_decision` is, from the same sample.

Alarms, all WARN, because a halted treatment run would be censored out of any comparison it is in:

- `imitation_ref_kl_high`: a regulariser's `kl` above `imitation.alarms.ref_kl_warn` (1.0 nats).
- `imitation_lambda_saturated`: `lambda_at_max` for `imitation.alarms.lambda_saturated_patience`
  (10) iterations. The reward is pulling harder than the anchor can hold.
- `actor_handoff`: in the first `warm_start.alarms.handoff_window` (20) unfrozen iterations,
  `ppo/kl` above `warm_start.alarms.handoff_kl` (0.05) or `ppo/clip_fraction` above
  `warm_start.alarms.handoff_clip` (0.3). The existing backoff acts on its own; this says why.
- `critic_unready` (19.5).
- `kl_dead` does not fire on a frozen iteration: its key is absent there.

Their thresholds live in `alarms`, with the other thresholds, so they stay out of the identity.

The ratio guard of section 7.7 predicts the importance ratio's arithmetic floor from
`net.noop_bias`, which is a seeded actor's largest probability. With an init, preflight defers it,
and once the weights are loaded it is measured instead: p_max and the largest legal |logit| over
the artifact's probe rows, refused above `ppo.ratio_atol` and warned within a factor of four. A
cloned actor's no-op probability is its own.

### 19.10 Demonstration shards (L4)

`royaleimitate.shards`. A shard directory holds `manifest.json` and part files of
`rows_per_part` rows (4,096 by default: a part is decompressed whole when it is read, and its
spatial column is the size of the planes times two bytes a row), one compressed array per column:

| Column | Type | What |
| --- | --- | --- |
| `spatial` | uint16 | value × 1000 |
| `card_ids` | uint8 | only when the env emits card-identity planes |
| `vector` | float16 | the codec's own rule for the vector |
| `mask` | uint8 | bit-packed, little bit order, `n_actions` bits |
| `action` | int16 | the label |
| `weight` | float32 | the row's weight in the loss |
| `flags` | uint32 | bits named in the manifest |
| `group` | uint64 | the match the row came from |
| `seat`, `tick` | uint8, uint32 | |
| `reward` | float32 | the env's reward for the row |

**Stored exact, quantised at load.** The writer refuses a spatial value that is negative, above
65.535, or not reproduced bit for bit by `float32(q) / 1000`, naming the plane, the tile and the
tick. The reader puts every batch through the target run's codec, pack then unpack, so a cloned
actor sees bit-identical inputs to the ones PPO's rollout and update see, and the shards survive a
change of codec table. A packed row carries no static plane (the codec supplies the run's own on
decode), so the reader checks each part's static planes against the run's and refuses a part from
another arena rather than show it the run's.

**Flags.** Bits 0 and 1 are reserved (`projected`, `other_command`). A producer names its own
from bit 8 up; the writer refuses a row carrying an unnamed bit.

**The manifest** records the environment's `config()` without its truncation, `card_names`, the
obs and action digests, `n_actions`, the engine's digests and parameters, the package versions and
commits, the producer's own parameters (free-form), every part file's sha256 and row count, the
flag names, and counts per flag and per split. It is written last, so a directory without one is a
write that did not finish, and the reader refuses it. A part whose bytes no longer match the
manifest is refused when it is read.

**The engine key** is the first sixteen hex of the sha256 over `{binary_sha256, build_digest,
calibration_digest, catalogue_sha256, obs_digest, engine_params}`, and names the directory.
`engine_params` is the engine's own constructor state as its `config()` states it, minus the card
list the catalogue already covers: what the calibration digest does not see, a card level or a
path search. `ShardContext.of_config(run_config)` builds a run's environment once and reads all of
it, so the writer and the reader compute the key with one function.
The reader refuses a catalogue, observation or action-space mismatch by name, always. It refuses an
engine-key mismatch unless the caller passes `allow_engine_mismatch` with a reason; the reason is
carried into anything trained from the rows. Rows replayed on another engine are a different
dataset, and the refusal is what stops that being silent.

**The split** is by group: a row is validation when the first eight bytes of sha256(group) read as
an integer are 0-4 mod 100. By match rather than by row, because rows of one match are not
independent.

`frame_stack` above one is refused by the reader: a shard row is one frame.

### 19.11 Where demonstrations come from

Demonstrations come from any source that yields shard rows (19.10). Replaying recorded logs
through the environment is out of scope here.

### 19.12 Behaviour cloning, the field reference and `demo_bc` (L5)

**Recording**, `royaleimitate.record(learner, teacher, out, battles=, seed=)`: a teacher -- a
scripted bot by name, an `Opponent`, or any policy such as a saved bot -- plays both seats of
battles in the learner's own environment, and every decision it makes is a shard row, grouped by
battle.

**Cloning**, `royaleimitate.clone(learner, demonstrations, out, epochs=, batch_size=,
learning_rate=, weight_decay=, patience=, seed=)`, trains the run's own actor, built by the run's network factory from its
`net`, so its `arch_digest` is the run's. No critic is trained.

- **Loss**: cross-entropy of the labelled action under the masked distribution
  (`MaskedCategorical`), weighted by each row's weight.
- **Split**: by group (19.10), so a battle is wholly in training or wholly in validation.
- **Optimiser**: AdamW with a cosine learning-rate schedule, and early stopping on validation NLL
  (the best epoch's weights are kept). Defaults: 20 epochs, batches of 256, learning rate 3e-4,
  weight decay 0.01, patience 3.
- **Output**: an actor artifact (19.2) with up to 1,024 validation rows as its probe rows and their
  log-probabilities, computed by the self-test's own function on the weights as saved. Its
  `meta["clone"]` records the validation NLL before training and after each epoch. `clone` returns
  the folder's digest.

**`royaleimitate fit-field-reference --rows <file.npz> --fields a,b,c --out <folder>`** fits the
`field_mlp` reference: weighted binary cross-entropy of a play label on named field columns, CPU.
The rows file holds one column per field (`[N, width]`), `label`, `weight` and `group`; validation
is by group as in 19.10. It reports NLL, Brier score and AUC on validation. Each input is rounded to
float16 before fitting, which is the codec's rule for the vector, so the model is fitted at the
precision a run reads it at. `--hidden`, `--epochs` and `--seed` set the fit; they, the metrics and
the rows file's sha256 go into the artifact's `meta`.

**`demo_bc`** is a regulariser: the BC cross-entropy on a batch of demonstration rows, added to the
actor's loss each minibatch with a scheduled coefficient. Its shards must carry the run's engine
key; there is no mismatch escape here.

### 19.13 Export and evaluation (L6)

RoyaleLearn's `docs/harness-spec.md` section 19.13.

### 19.14 What stays out of the public packages

The logs, the shards, the fitted references, the cloned weights and any run initialised from them
live where their owner keeps them, under a path the config names with a digest. Nothing in this
package reads a log format.

### 19.15 Controls, each seen failing on a plant before it is trusted

- **The blind control (L1).** The block present with every coefficient zero, no init and a scale of
  one reproduces the run without the block bit for bit for three iterations: every weight, every
  optimizer moment and every action. The reference is another seed's actor, so the KL it reports is
  not zero and the term's whole path ran. Plant: a coefficient that leaks a thousandth past zero
  must make it fail. (A draw from a named stream cannot contaminate this harness: every stream is
  addressed by name, so an extra draw moves nothing else.)
- **The init-identity control (L1).** An init from an artifact holding the seeded weights, with no
  freeze and no regulariser, reproduces the run without the block bit for bit for three iterations,
  state digest included. Plants: an init that also touches the critic must break it, and one weight
  changed in the artifact must make the self-test refuse.
- **The chain-rule identity (L2).** `kl_noop + kl_card + kl_tile == kl`. Plant: dropping the p_ref
  weighting of `kl_tile` must break it.
- **The shard round trip (L4).** Rows written and read back through the codec equal the rows the
  codec produces from the env's observation directly. Plant: a spatial value off by one quantum.
