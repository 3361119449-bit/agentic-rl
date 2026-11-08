# Grounded Judge v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Correct verified Judge semantic/label/attribution faults while preserving every reward formula and runtime default.
**Architecture:** Explicit grounded-v3 configuration opts into atomic current-turn audits and witnessed global attribution. Legacy v2 records and schemas remain supported. A normal current-turn request is made once; only contradictions/global-neutral conflicts consume bounded review attempts. Reconciled Judge results are audited before unchanged reward scoring.
**Tech Stack:** Python, Pydantic2, existing cached httpx client, pytest, CPU Tau2 replay.
**Spec:** /root/autodl-tmp/judge_target_v3_20261108/SPEC.md

## Global Constraints
- Preserve progress_scale0.5, ProCreditv4/Target semantics and all existing sign guards/formulas.
- Only new judge.grounding_version=judge-grounded-v3 enables new outputs. Legacy caches/configs remain reproducible.
- Preserve35-turn cutoff zero/retries, BF16, concurrency8 and server fulltool/eval defaults.
- No reference-action safety or duplicate-query changes, no GPU generation, no source trajectory edits.

## Review Focus
- Claimed current-tool outcomes are future evidence; reject leaked prefix references.
- Missing facts without definite false assertions are neutral, not fabricated facts.
- Financial units, passenger counts, refunds and preserved-segment pricing must not be conflated.
- A global-neutral conflict cannot silently discard genuine violations or mutate raw Judge provenance.
- Frozen older records cannot be accepted as new structured assessments or reused from old cache identities.

## Task 1 Atomic current-turn audits
Files: new reward/grounded.py and judge/grounded.py; extend reward/simple_turn.py and client.py; tests/test_grounded_judge_v3.py.
Interfaces: build_turn_inputs(..., judge_version=...) defaults v2; v3_request/validate_grounded_assessment -> legacy compatible derived decision; structured trace saves raw assessments.
- [ ] Write failing tests for derived bad labels, neutral/refusal controls, scoped reason contradictions, arithmetic/ranking/authorization mismatch, causal evidence and cache separation.
- [ ] Run red tests and save outputs.
- [ ] Implement minimal typed assessments/validation and grounded prompt; bounded client validation feedback without altering legacy wire behavior.
- [ ] Run focused then full suite; commit task.

## Task 2 Witnessed global attribution and integration
Files: schemas.py, judge/grounded_policy.py, judge/prompts.py/client.py, runtime actor/scoring_retry/record validation, active3config versions and preservedv12snapshot.
Interfaces: witnessed global result carries grounding_version and per-criterion actor/policy quotes; judge_grounded_turns returns decisions, assessments, reconciled result and provenance; finalize trace consumes unchanged computed policy_credit.
- [ ] Write failing tests for exact witnesses, causal citation, latest confirmation, old-error noncontagion and conflict review outcomes; ensure deterministic safety remains.
- [ ] Implement global prompt/schema and targeted conflict reconciliation before unchanged score_trajectory, wire actor/retry/config and cache fingerprints.
- [ ] Verify structured record replay/provenance and old record compatibility; full suite and native relevant tests; commit task.

## Task 3 Real frozen-data validation and release
Files: docs/grounded-Judge report, regression fixtures derived from48cases, offline-only scorer tooling.
- [ ] Test three known conflicts and48 targeted prefixes against actual authorized DeepSeek API, saving request/result/cache identity and outcomes; revise semantic instructions where evidence demonstrates failures.
- [ ] Re-score frozen80 into separate output, cutoff0/noJudge path intact; no rollout. Recompute using unchanged native progress/official scores and advantage formula.
- [ ] Complete full suite/lint, fresh whole-branch review and necessary fixes with red/green evidence.
- [ ] Publish PR, pass CI, merge; server download merged GitHub ZIP, preserve local overlays, backup selected files, deploy and verify live tests/source hashes/SFT inventory.
- [ ] Deliver measured report and remaining semantic limits; never report manually curated labels as API results.
