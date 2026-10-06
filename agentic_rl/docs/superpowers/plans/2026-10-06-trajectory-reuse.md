# Preserve usable on-policy trajectories

The user authorizes all three fixes: salvage scored members after transient slot
exhaustion, consume all viable groups, and consume a sub-target remainder at the
three-wave cap. Fatal failures still stop training. Unscored, invalid-user and
failed-session outputs must never contribute. Policy violation turns stay negative.

1. Verify regressions against the production sampler and shared credit code.
2. Publish settled session membership from the failure-aware worker. Accept
   partial v4 groups only with exact terminal transient-failure evidence; preserve
   original uid, task, policy, checkset and initial-state identities. A singleton
   can use genuine local credit but has zero scalar group advantage.
3. Consume every signal-bearing group after draining each wave; at the cap use
   any remaining viable groups. Never stash surplus for a later policy version.
4. Carry terminal membership through veRL balancing. Ignore zero-mask synthetic
   padding in baseline, audit, metrics and gradients; retain original log-probs.
5. Update replay/config/docs and counts. Run full tests, lint and diff checks.
   Commit locally; do not push without a new push request.

Validation: missing members and fatal failures fail closed; failed-session output
cannot be salvaged; seven-member and singleton groups retain negative local credit;
all-zero groups still skip; CPU Torch checks both nested and dense padded batches.

Completed: regressions initially reproduced the fixed-size member/group loss and
missing worker membership; all implementation steps are complete. Final full
suite: 695 passed, 25 skipped. Ruff and diff checks passed. Independent final
review found no actionable issues. Real GPU/API smoke remains unverified.
