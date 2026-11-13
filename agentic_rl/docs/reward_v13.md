# Reward v13: immediate progress and independent refusals

This is the user-approved credit-assignment change, based on main
`d10a8e7622447b28df3021ae5ec80a7f6c855301`. It does not replace ProCredit v4.

## Current default and compatibility

`configs/rl/airline_procredit_v4.yaml` now selects:

```yaml
project:
  reward_version: v13.1-procredit-immediate-independent
reward:
  check_reference_write_scope: false
credit:
  version: procredit-turn-v4
  progress_credit: immediate_delta_v1
  turn_centering: none
  final_turn_rule: simple-turn-v2
```

The prior default is frozen in `airline_procredit_v4_v12.yaml`. Missing
`progress_credit` retains the historical remaining-return algorithm; `simple-turn-v1`
retains its old refusal deduplication/revocation semantics and audit identities.
New and historical label versions cannot silently mix. Start a new run/tag rather
than claiming exact continuation under the old reward identity.

v13.1 disables the reference write-scope check in the active ProCredit profile.
Required actions still contribute to task completion and progress. An authorized
free-baggage update can receive normal turn credit when the reference lists the
bag as part of a new booking. Explicit policy violations and failed tools with
database effects still contribute to local penalties. Historical configs and
frozen scoring inputs without this setting retain their original behavior.

## Trajectory completion checks

The active training and official-test profiles use the reviewed `v2` semantic
annotations. Each omitted criterion records its existing DB or required-action
coverage in `covered_checks`. The train rubric goes from 73 to 40 criteria, and
the test rubric from 50 to 19. The original `v1` files remain available for replay.

Official COMMUNICATE checks find literal text. Residual checks still verify the
meaning of an amount, a policy explanation, or any behavior DB and REQ do not
establish. Where all semantic criteria are covered, Judge Semantic is inapplicable.
The other applicable components still determine task completion. The global Judge
receives only the residual semantic criteria and continues to score policy and
transfer. The per-turn Judge still uses its own existing rubric.

## Formula and precedence

For nontruncated trajectories, keep the score `R = T + 0.5 * phi_final` and the
population-standardized trajectory advantage `B = (R - mean(R))/(std(R)+1e-6)`.
For each assistant generation, compute:

```python
local_progress = 0.5 * (phi_after - phi_before)
a = B + local_progress - policy_cost - process_cost
if policy_bad:
    a = min(a, -1.0 - process_cost)
if bad_turn:
    a = min(a, -1.0)
elif eligible_correct_refusal:
    a = max(a + 1.0, 1.0)
```

There is no remaining-return term, turn mean, turn-level centering, added success
bonus or repair bonus in the new path. Negative progress is retained, not clipped.
The exported legacy `turn_mean` field is zero for interface compatibility, not a
computed baseline. `turn_returns` contains immediate scaled deltas in this mode.

**Zero truncation wins over all of these rules.** Actual cutoff reasons yield zero
scalar, turn and token credit, even in mixed groups. Natural completion on turn 35
is not a cutoff. Observation/padding token credit remains zero. The 35-turn limits,
bounded training cutoff resampling (at most two), and official evaluation accounting
are unchanged. See [TRUNCATION_35.md](../TRUNCATION_35.md).

## Refusal eligibility and Judge

Every clean, correctly grounded policy refusal is independently eligible, even
if the same request was refused before. A later violation never revokes a past
award. A turn earns at most one bonus. Any current policy, semantic, native tool
or deterministic duplicate-query error wins over a refusal in that turn.

The existing global Judge plus one prefix-bound per-turn Judge are retained.
Only the per-turn eligibility instructions/cache identity change. The v2 request
includes the actual preceding conversation, but no previous Judge decisions.
The existing flat `revoked_refusals` field must be empty in v2; its schema is kept
for historical compatibility. Current-target evidence binding and bounded retry
validation are unchanged. There is no numerical matching rule or required
`calculate` call. Semantic arithmetic and policy checks remain the Judge's job.

Repeated rewarded refusals can favor longer exchanges; this is an accepted design
tradeoff, not evidence of a failure in this implementation. Existing penalties,
cutoff policy, query repetition rule and optimizer parameters were not retuned.

## What this formula guarantees, and what it does not

For nontruncated samples, merged bad turns are at most -1, and eligible clean
refusals are at least +1. Original stronger policy penalties are never weakened.
A normal positive progress increment **does not guarantee positive final credit**:
negative trajectory B can outweigh it. No extra progress floor was requested.
Wrong Judge labels still yield wrong sign constraints; arithmetic passing does
not establish semantic label accuracy or a trained model's behavior improvement.

## Offline replay

From `agentic_rl/`:

```bash
PYTHONPATH=src python scripts/replay_immediate_credit.py \
  /path/to/extracted_v11_bundle /path/to/empty_output \
  --review-labels /path/to/review_labels.json
```

This diagnostic script compares the original v13 credit assignment on the frozen
80-trajectory/1227-turn bundle. It retains the complete archived semantic decisions
for that comparison. It performs no API calls or training. It first
reproduces archived v11 scores/advantages, then compares the current-main v12
formula against v13 using identical saved decisions, task scores, progress,
costs, groups and token mappings. New default 35-turn zero policy is applied to
**actual recorded cutoff reasons**; it does not fabricate five extra turns for
historical 30-turn hard stops.

Optional review labels export two additional comparisons: current-prefix local
AI labels with original global policy retained, and an explicitly named full
AI-policy numerical counterfactual. The latter is not a newly scored production
record. This avoids attributing a Judge correction to a formula change.

The source has four samples per task; online configuration remains group_size 8.
Do not treat the offline B values as actual online training tensors. All source
records are hash-checked and unchanged; outputs require a new empty directory.

## Validation on the frozen bundle

The candidate reproduces all 80 archived scores and all 1227 archived turn
advantages. With saved semantic decisions held fixed, v12-to-v13 changes 585 turns;
311 active bad turns remain negative, 31 current clean-refusal awards are positive
(v12 had 15 after zero-cutoff exclusion). Six recorded cutoffs contain 165 turns,
all zero. Original 92,475 actor tokens and observation locations are unchanged.

Seven trajectories (115 turns) were additionally read and judged by the assistant:
Task22/s3,18/s0,30/s0,25/s3,6/s0,32/s0,24/s3. Key findings:

- Task22/s3: correct cabin/passenger/baggage progress gets +0.125,+0.125,+0.25;
  equal-score normal preparation and final report become 0.
- Task18/s0: each of three correct writes gains 1/14 but remains about -0.548612,
  because B is about -0.620040. This is the specified formula, not a sign guarantee.
- Task6/s0: three independent clean insurance refusals each receive +1.
- Task30/s0: an erroneous quote becomes -1 under AI review; with the old neutral
  label it remains +1.732048. The correct write is +2.232048 in either case.
- Task32/s0: AI-reviewed correct refusal is +1 despite the later violation at -1.
  Retaining the old global policy mislabel still makes that refusal -1; production
  Judge topology was not changed to hide that limitation.

These are offline checks and assistant judgments, not independent human gold,
new API Judge predictions, GPU optimization or improved task success rates.
