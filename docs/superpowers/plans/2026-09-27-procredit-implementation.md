# ProCredit Reward Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Track results below; do not start GPU or paid API jobs automatically.

**Goal:** Implement the approved strict-terminal/progress/turn-credit design end to end while preserving legacy training and official evaluation.

**Architecture:** Keep deterministic progress and advantage math independent of veRL. Record delivered prefix boundaries and token turns in the actor, then use one shared group-credit calculation in filtering and the trainer. Version new reward records and bind the new configuration to resume identity.

**Tech Stack:** Python 3.12–3.13, Pydantic, pytest, fixed Tau2 and veRL v0.9.0.

**Spec:** ../specs/2026-09-27-tau2-procredit-reward-design.md

## Global Constraints

- c=0.5, gamma=1, turn coefficient=1, population std, epsilon=1e-6.
- S=V*max(0,m*(R+0.5*Phi_T)-P); m=0.75 on existing truncation reasons; score ceiling=1.5.
- Center only valid nonempty policy turns; gate turn advantage after centering.
- Group by real sampling uid; 8 trajectories/group, 4 groups/update, at most 3 batches of 8 groups.
- Preserve official reward/pass metrics, old reward mode, token IDs and rollout old log-probs.
- Do not alter the approved 24/6/20 split, agent prompt or simulator policy.

## Review Focus

1. Constant terminal scores can still have useful turn credit: exercise real filter and actor-bound tensors.
2. Rejected/truncated text must not pass COMM checks: exercise production actor prefix capture.
3. API/scoring failure must never become gate=false: cover frozen scoring retry and incomplete groups.
4. Reward >1 must survive schema, persistence, queue and trainer; legacy schema still rejects >1.
5. Same task in different prompt groups must not merge; group identity and offline membership must be immutable.

## Task 1: mathematical contract and reward mode

Files: new advantages.py and tests/test_procredit_credit.py; reward/score.py, schemas.py, tests/test_procredit_reward.py.

Interfaces: CreditConfig.from_project(project), compute_group_credit(rows, config) returns per-trajectory trajectory/turn/token advantages and has_signal; score_trajectory(..., progress_trace=None) accepts explicit new-mode progress.

- [x] Write hand-calculated tests for constant-score turn signal, unequal lengths, invalid gates, empty checksets, token masking and malformed identities.
- [x] Run new tests and confirm missing behavior fails.
- [x] Implement validated credit data and math; add strict_progress_v1 score branch without changing legacy score.
- [x] Verify new tests and legacy reward tests.

## Task 2: deterministic progress and prefix capture

Files: new reward/progress.py and tests/test_procredit_progress.py; environment/tau2_gym.py; agent_loop/airline.py and tests/test_procredit_rollout.py.

Interfaces: build_progress_trace(task, messages, prefix_lengths, events, required_actions, dependencies, transfer_rule, initial_state_fingerprint, evaluator=None) -> validated trace; production evaluator uses pinned Tau2 in separate environments. Trace contains fixed checks, initial values, per-turn bits and Phi.

- [x] Write tests for initial-check exclusion, DB regression, COMM prefix-only evidence, REQ dependencies and empty sets.
- [x] Verify failures, then implement deterministic trace builder and lazy Tau2 evaluator.
- [x] Write production-loop tests for rejected and truncated turns, then capture raw delivered prefixes and response_turn_ids.
- [x] Verify prefix recording cannot include cleanup or mutate interaction data.

## Task 3: queue, bounded sampling and actor advantages

Files: new procredit_runtime.py, tests/test_procredit_runtime.py; verl_capped_trainer.py; contract_tests/test_real_verl.py.

Interfaces: runtime group classifier consumes extra_fields.procredit and actual queue uid; trainer credit writer materializes token advantages from the same pure calculation. Atomic group audits retain original membership.

- [x] Write integration tests proving equal-score groups survive and zero-signal/partial-failure groups do not.
- [x] Verify failures, implement bounded classifier that disables upstream scalar constant filtering only for the new mode.
- [x] Write trainer-path tests for actual nested/padded tensor writing and frozen old log-probs.
- [x] Implement _compute_advantage override and group audit; run contract tests available locally.

## Task 4: configuration, frozen retries, offline scoring and identity

Files: new configs/rl/airline_procredit_v1.yaml; training_config.py, config.py, rl_resume.py, scoring_retry.py; scripts/rescore_saved_trajectories.py and new scripts/rescore_procredit_groups.py; corresponding tests.

- [x] Write configuration tests for mixed legacy fields, disabled Judge, wrong filter or credit values.
- [x] Verify failures; implement dedicated new config, startup validation and code/credit/check identity.
- [x] Write replay/resume tests; pass frozen progress to retries and per-record rescoring.
- [x] Add exact-membership offline group rescore with new-output-only semantics.
- [x] Verify official evaluation and legacy launchers remain unchanged.

## Task 5: verification, documentation and independent review

- [x] Run the full CPU suite and targeted real dependency contracts where available; record skips and missing dependencies.
- [x] Run static checks and whitespace checks.
- [x] Document the new command, audit fields, resume incompatibility and GPU acceptance remaining.
- [x] Request one fresh-context whole-change review, fix substantive findings with regression tests.
- [x] Leave all implementation artifacts reviewable on the feature branch; no remote push or GPU job.

## Execution ledger

- Baseline: 378c062; design approved by user instruction “请实施”.
- Ruling: execute inline in the existing available checkout on codex/procredit-reward; native worktree tool unavailable and no separate concurrent checkout work is present. This preserves the user's workspace location.
- Ruling: user's implementation instruction authorizes continuing through routine execution steps without another design/plan approval pause.
- Baseline CPU suite ran successfully with base Python 3.13; optional torch/transformers coverage skipped (24 skipped markers). Full final counts will be recorded explicitly.
- Environment: local base Python has core project dependencies, no Tau2/veRL or GPU training stack. Do not represent mocked integration tests as real GPU acceptance.
- Ruling: retain the progress ledger in this plan instead of shell-specific temporary skill scripts, so progress survives Windows tooling and context compaction.
- Pre-flight shared interfaces: Task 1 score/credit fields feed Tasks 2–4; `phi` has T+1 entries, raw policy turn indices are zero-based, and queue membership is the actual uid/session key. Producer and consumer tests cover this contract.
- Tasks 1–4 implemented with failing tests before behavior changes; legacy reward, truncation, rollout, configuration and resume regressions passed throughout.
- Verification before final review: base CPU 484 passed / 27 skipped; after record integrity additions, CPU with real PyTorch 2.9.0 and TensorDict 0.14.2: 490 passed / 25 skipped (18.43s). Optional real Tau2 contract: 1 skipped (Tau2 absent).
- Ruling: run real tensor tests with existing local Torch plus isolated validation tools under ignored outputs; full pinned veRL/Tau2 integration remains an explicit training-environment acceptance item. Cost if wrong: interface incompatibility may still require adjustment before the first GPU update.
- Ruling: preserve the approved 24/6/20 split by rejecting `full_train` only in the new reward mode; legacy full_train remains unchanged. Use internal_dev for the 24-task experiment. Cost if wrong: users needing 30-task final training must authorize a separate experiment.
- Source compatibility inspected against exact pinned upstream replay-buffer, trainer, gym, message and evaluator sources. No live simulator/API/GPU jobs started.
- Record hardening: altered turn maps, scalar scores and checkset bindings reproduced as 3 failing tests, then fixed through the shared actor/offline record adapter (5 rollout tests passed).
- Static check found one pre-existing extra blank line in test_run_isolation imports; removed only that blank line to allow the full configured lint command to pass.
- Final static verification before review: Ruff passes across src, scripts, tests, sft_contract_tests, contract_tests and training/qwen3_4b_sft; git diff --check passes. Training, single-record rescoring and group rescoring CLI help all work. Actual tensor/metadata adapter tests separately pass 6/6.
- Fresh-context read-only reviewer dispatched using requesting-code-review; review covers uncommitted modifications and new files. Review package retained under ignored outputs/procredit-review-package.md.
- Final review: no Critical or Minor findings; one Important finding confirmed. Parent-map/deletion Hydra overrides could disable the trainer ProCredit flag while leaving actor reward mode enabled.
- Final: fixed mode mismatch / protected override bypass — test_cli_cannot_turn_off_actual_credit_hook_or_change_discount and test_composed_trainer_mode_must_match_actor_reward_mode reproduced 3 failures before the fix; 18 passed / 3 optional-torch skips in the targeted base-Python run afterward. Both directions of mode mismatch now fail before selecting a trainer path. Protected parent-map/deletion overrides are rejected.
- Final full suite: `pytest tests -o addopts= -q -p no:cacheprovider` with real local Torch/TensorDict → 493 passed, 25 skipped in 16.56s. Full Ruff scope and git diff --check pass. Logs retained under ignored agentic_rl/outputs/procredit-final-tests.log and procredit-review-red.log.
- Final: Ruling: reviewer could not judge full pinned veRL/TransferQueue/Tau2 transport — retain explicit real-dependency contracts and require their execution in the training environment, with no claim of local integration acceptance. Cost if wrong: first live integration may require a compatibility fix.
- Final: Ruling: reviewer could not judge GPU loss consumption, checkpoint recovery or effectiveness — no API/GPU run was authorized as part of this implementation plan; retain these as documented experimental acceptance tasks. Cost if wrong: no evidence yet that the new reward improves training outcomes.
- Final: Ruling: use the finishing-a-development-branch workflow's keep-as-is outcome, consistent with this plan's feature-branch delivery scope; preserve all edits and documents without commit, push, merge or cleanup. Cost if wrong: user must choose a later integration action.
- Deferred minors: none. No second review dispatched; the confirmed fix was verified by RED→GREEN regressions and the complete final suite.
- Follow-up: user explicitly requested “请提交”; commit the implementation, tests and documentation on codex/procredit-reward. This supersedes the earlier keep-uncommitted delivery decision; no remote push requested.
