# Immediate credit implementation plan

> Execute inline using executing-plans and TDD; the user supplied and approved the spec.

Goal: implement immediate progress credit and independent per-turn refusal rewards.
Architecture: add explicit selectors to existing ProCredit v4, retain historical paths.
Tech: Python, Pydantic, pytest, existing GitHub CI.
Spec: docs/superpowers/specs/2026-11-10-immediate-credit-design.md

## Constraints
No changes to scalar task reward, progress checks, penalties, transfer/duplicate rules,
Judge layers, optimizer, rollout settings or the 35-turn zero/retry policy. No GPU or
paid model API calls. Frozen input files are read-only. Preserve observation masks.

## Review focus
- Historical labels/configs cannot silently change meaning.
- Repeated or later-reversed clean refusals keep their earlier independent credit.
- Policy/process errors override refusal; existing stronger negative values survive.
- Cutoffs remain zero even in mixed groups and with otherwise-positive labels.
- Delta indices map to the originating assistant generation, not observation tokens.

## Task 1: immediate progress
- [x] Write independent expected-value tests for equal/mixed groups, regression,
  policy/process costs, observation mapping, legacy output and zero truncation.
- [x] Run and observe RED; implement CreditConfig.progress_credit and delta path.
- [x] Run focused tests GREEN and full suite; record evidence.

## Task 2: independent refusal
- [x] Write RED tests for repeated rewards, no retrospective revocation, bad precedence,
  one reward per generation, cache isolation and actor/retry/runtime wiring.
- [x] Implement versioned simple-turn-v2, no prior refusal context or revocation,
  existing flat Judge schema with empty revoked_refusals, and causal evidence guards.
- [x] Run focused and full tests GREEN; preserve legacy v1 fixtures.

## Task 3: integration and delivery
- [x] Snapshot exact v12 YAML; opt default into v13 selectors; validate compatible
  config combinations; document formula and unchanged modules.
- [x] Replay frozen trajectories, compare old/new formulas at fixed labels; independently
  review several complete real trajectories and calculate every turn at those labels.
- [x] Separate-pass self-review (no subagent tool); fresh full-suite and replay checks.
- [x] Publish tested changes on a new branch and open PR; do not merge or train.
