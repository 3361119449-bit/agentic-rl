# Optional post-completion regression credit

Implement the user's supplement on `0c8ee42` without changing scalar rewards,
official evaluation, group baselines, token alignment, sampling or PPO.

- `credit.post_completion_credit: null` by default;
  `post_completion_regression_v1` requires ProCredit v4.
- Use all official DB/COMMUNICATE checks, including initially satisfied checks,
  from verified prefixes. Never use phi or required-action completion as proof
  of official success. Unsupported reward bases or user-side DB tools disable
  the mechanism rather than guessing.
- Only an observed true-to-false DB transition after full completion, caused by
  recorded Assistant mutation(s) with a proven DB effect, is punishable. A turn
  containing several tools has one shared sign restriction. Unobserved success
  inside a single turn cannot earn a completion bonus.
- Only trajectories with such a regression can earn the first completion bonus.
  Initial full success earns no bonus. No repeat bonuses after repair. Recovery
  may succeed officially; its earlier destructive turn still gets the restriction.
- Completion and DB subgoal positive credit share a trajectory cap of 0.5.
  Keep existing subgoal credit; give only the remaining budget at the first safe
  full-completion turn. No bonus on policy or mixed wrong-write turns.
- User explicitly requires the first safe completion turn's FINAL advantage
  to be strictly positive. After all local/process/trajectory terms combine,
  apply max(existing, max(1e-6, 0.5-process_cost)) once on that turn, even when
  the DB reward budget is exhausted. This sign restriction does not add a
  second goal reward; record its separate adjustment, which can exceed 0.5
  to offset negative group advantage. Existing policy ceilings have priority.
- Apply the regression token ceiling `-1.0 - process_cost` after combining task
  and local terms. Combine with the existing policy ceiling by minimum, without
  subtracting an extra policy-sized penalty for the same destruction.
- Save a versioned completion trace; live scoring, frozen retries and record
  validation use one builder. Turning off the flag preserves old audit output.

Work: tests first; small scoring/record/config/advantage hooks; standalone offline
audit reusing existing fixed-tool harness; docs with Task 12 T6/T10 comparison.
Audit all original 738 records/96 groups, compare baseline/post and DB/both modes,
check shared caps, policy/process, zero masks and every changed turn; replay
candidate DB transitions against the fixed Tau2 database and tool bodies.

User subsequently requested no real PPO smoke; they will run it later.

Completed: 738 original PPO records / 96 groups; 56 confirmed DB regressions
(52 initially complete, no completion bonus), 3 bounded completion bonuses,
689 independently replayed DB prefixes, no double credit or mispenalties.
Real Task 12 T6 is protected at final advantage +0.5; its T10 destruction stays
negative and official failure is unchanged. Full validation results and all
eligible completion-turn sign checks are recorded in the machine audit report.
No GPU job, commit or push was performed during this implementation.
