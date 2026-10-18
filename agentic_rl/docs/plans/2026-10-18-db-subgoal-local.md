# Optional database subgoal local credit

Based on the user-supplied implementation plan and baseline `967e3b5`.
Purpose: give independently correct reservation updates local credit without
changing official success, scalar reward, sampling budgets, or PPO.

- Add `credit.db_subgoal_credit: null` (default); opt in with
  `db_subgoal_local_v1`, only with ProCredit v4.
- Eligibility requires a DB task, at least two distinct supported updates on
  one reservation, no dependent or unsupported required mutations. Cancel/book
  transactions and compensation targets are excluded by the generic rule.
- Capture the initial reservation from the live DB without adding tool calls.
  Use successful tool-result snapshots in execution order afterwards. Require
  both current fields and the existing full action matcher, including payment.
- A first completion earns `0.5/N` only on its own clean turn. A mixed or policy
  violating turn earns no new positive credit. Never transfer withheld credit
  onto a later read or unrelated action. Initial satisfaction cannot earn credit.
- Verified regressions lose `0.5/N` on the destructive turn, at most once per
  goal (total regression cost <= 0.5). Unknown state is diagnostic, not proof of
  a regression. Correct earlier credit is retained; recovery cannot farm credit.
- Keep `progress_trace.phi`, task returns and all group baselines unchanged.
  Add local values before the existing policy sign restriction, preserving
  process costs, observation masks and padding. Disabled audits remain identical.

Implementation order:
1. Failing tests for state, action/payment matching, mixed writes, regressions,
   repeats, initial satisfaction and config-off identity.
2. `reward/db_subgoals.py`, optional record trace, frozen initial inputs, scoring
   and retry integration; pass verified trace through `credit_from_record`.
3. Optional local addition in `advantages.py`, config validation/default and docs.
4. Offline audit CLI: preserve original group membership, reproduce baseline
   token advantages, verify snapshots with fixed Tau2, recompute enabled credit,
   save differences and every identifiable wrong-write sign transition.
5. Run focused/full tests and lint, audit all 738 PPO records / 96 groups, inspect
   the six named real trajectories and task 42. Do not start RL experiments.

Review focus: wrong payments despite correct fields; stale snapshots after a
write; mixed-turn attribution; initial/repeated completion; frozen retry parity.

Completed: 738 records / 96 original groups, exact disabled replay, no newly
positive wrong-write turns; 848 tests passed, 30 skipped. See
`DB_SUBGOAL_LOCAL_V1.md` and `data/audits/db_subgoal_local_v1_967e3b5.json`.
No training was started during implementation.
