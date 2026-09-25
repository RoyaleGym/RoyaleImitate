# The imitation sections' alarms

A run with an `imitation` section holds the regularisers' two alarms; a run whose `warm_start`
section schedules the actor's learning-rate scale holds RoyaleLearn's two freeze alarms, with the
thresholds under `warm_start.alarms`. RoyaleLearn's `docs/running.md` section 3.6 describes the
freeze alarms. What follows is RoyaleLearn's former section 3.6 as it stood when the code moved,
covering all four.

## Learning from demonstrations (four alarms)

These fire only on a run with a `warm_start` or `imitation` section: a run that started from a
cloned policy, or that is held near a reference policy while it learns. Section 19 of [spec.md](spec.md) describes the sections. All four warn and none stops the run,
because a run stopped early would drop out of any comparison it is part of.

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `imitation_ref_kl_high` | `imitation/<name>/kl` | a regulariser's KL above `imitation.alarms.ref_kl_warn` (1.0 nats) | warn, 1 |
| `imitation_lambda_saturated` | `imitation/<name>/lambda_at_max` | the anchor's coefficient sat at its ceiling | warn, `imitation.alarms.lambda_saturated_patience` (10) |
| `actor_handoff` | `ppo/kl`, `ppo/clip_fraction`, `ppo/iterations_since_unfreeze` | in the first `warm_start.alarms.handoff_window` (20) iterations after a frozen actor starts to move, KL above 0.05 or clip fraction above 0.3 | warn, 1 |
| `critic_unready` | `ppo/ev_at_unfreeze` | the critic's explained variance when the actor was unfrozen, below 0.3 | warn, 1 |

**`imitation_ref_kl_high`.** The policy has moved far from the reference it is anchored to. That
can be the anchor letting go on schedule, or the reward pulling the policy somewhere the reference
never goes. Read `imitation/<name>/kl_noop`, `kl_card` and `kl_tile` to see which part moved.

**`imitation_lambda_saturated`.** The coefficient has been at `coef.max` for ten iterations and
the KL is still above its budget. The anchor is pulling as hard as it is allowed to and losing.
That is a statement about the reward, not about the anchor.

**`actor_handoff`.** The first iterations after the freeze moved the policy fast. The
learning-rate backoff acts on its own; this says why it acted.

**`critic_unready`.** The critic was trained on the frozen policy's battles and still
explained little of the return when the actor was let go, so the first policy updates run on a
poor baseline. A longer freeze is the usual answer.

