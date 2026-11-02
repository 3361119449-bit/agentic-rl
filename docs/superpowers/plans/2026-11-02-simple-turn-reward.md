# Simple turn reward implementation plan

> For agentic workers: use superpowers:executing-plans to implement this approved specification task by task.

Goal: implement the user's 2026-10-31 simple reward report on main 08cc07a, retaining the complete original A0 calculation.

Architecture: causal, cached per-turn binary semantic decisions plus deterministic policy/tool labels; a single final advantage transform after all existing terms. Existing scalar rewards and GRPO means remain unchanged.

Tech stack: existing Python, Pydantic, HTTPX Judge client, veRL adapter, pytest.

Spec: user-provided simple_reward_plan_validation_20261031.txt. The user explicitly requested implementation and GitHub publication; no new design approval is needed.

## Global constraints

- bad: min(A0, -1); otherwise eligible refusal: max(A0 + 1, 1); otherwise A0.
- Preserve old policy attribution and strong negatives. Never add refusal bonus to scalar rewards.
- Use bounded actor-visible messages only, with zero-based message and assistant-turn indices.
- Each semantic request sees one causal prefix. Future commitments can revoke earlier refusal eligibility, never retroactively make its decision bad.
- Deduplicate by target plus prohibited operation; bad wins; no repair bonus or extra efficiency penalties.
- Observation and padding masks remain zero. Semantic failures retry scoring frozen trajectories; they must not silently become neutral labels or cause new rollout generation.
- Keep the old v4 configuration available. Publish a backup branch before updating main. No training, live evaluation, paid Judge requests or server deployment in this task.

## Review focus

Future/hidden evidence; malformed or missing labels; mixed bad/refusal turns; repeated refusals and unrelated later requests; frozen-scoring retry and token/padding alignment.

## Task 1: terminal advantage and label contracts

Files: reward/simple_turn.py, advantages.py, tests/test_simple_turn_reward.py.
Interfaces: transform_advantage(a0, bad_turn, eligible_correct_refusal); build_turn_inputs(messages, policy, turns); build_label_trace(inputs, decisions, events, policy_credit).
- [x] Write failing literal numeric cases from the report and causal/dedup/error tests.
- [x] Implement strict binary decisions, tool-error union, refusal eligibility, audit trace and final transform.
- [x] Verify old A0, scalar means, unlabeled turns and masked tokens remain unchanged.

## Task 2: live and frozen scoring integration

Files: judge/simple_turn.py, agent_loop/airline.py, scoring_retry.py, failures.py, procredit_runtime.py, tests/test_simple_turn_scoring.py.
Interfaces: score_simple_turns(inputs, events, policy_credit, judge) returns a validated trace; validate_record_labels(record) reconstructs and checks frozen evidence before training.
- [x] Test request prefixes/cache separation, transport retry, missing evidence, live/frozen completion and masks before implementing integration.
- [x] Reuse DeepSeekJudge with versioned prompt/schema/cache and existing concurrency budget.
- [x] Store inputs, decisions, reasons, evidence and old/new advantages; propagate trace into queue group audits.

## Task 3: configuration, documentation and publication

Files: v4 config plus legacy copy, config validation, README and simple reward guide.
- [x] Enable the new final rule for default v4; retain --config access to unchanged legacy mode.
- [x] Run complete CPU suite, lint and diff checks; obtain independent review and address substantive findings.
- [ ] Back up old main, publish tested commit and verify remote tree. Report limits and commands.

## Execution ledger

- Base: 08cc07a. Dedicated checkout work/simple-reward; no unrelated working changes.
- Ruling: use one causal semantic request per assistant turn, with earlier refusal decisions supplied as context. This enforces the report's no-future-evidence rule and costs additional cached Judge calls during future training.
- Ruling: binary labels are versioned metadata; legacy records/configs keep their original behavior. New mode fails closed on absent labels.
- Task 1: RED 18 failed/1 passed, then GREEN 19 passed; later queue-policy retention and current-generation revocation evidence checks also observed RED then GREEN.
- Task 2: RED three integration failures, then GREEN live/frozen transport tests. Actual PyTorch dense/nested and padding regressions run with the available d2l interpreter; no GPU execution.
- Ruling: existing offline rescoring remains supported for already labeled records but must reject unlabeled records under the new default; old historical tests explicitly choose the byte-preserved legacy config. No replay training adapter was added.
- Task 3 verification before review: complete CPU suite with PyTorch: 912 passed, 42 skipped (optional dependencies, one pyarrow DLL warning); Ruff and git diff --check passed. No paid API or live training/evaluation run.
- Independent review found two important refusal-identity boundaries. Both addressed in one fix pass: preserve request identity through deterministic bad repeated refusals, and represent all refused requests in a generation with request-specific revocations. Regression cases were observed RED then GREEN.
- Ruling: a turn can refuse multiple requests but receives at most one +1; record all keys for deduplication. Revoking one request leaves another first-valid refusal in the same earlier turn eligible. This preserves the two-boolean final rule while supporting multi-request dialogue.
- Final verification after all review fixes: PyTorch CPU full suite 919 passed / 42 skipped, base interpreter supplemental suite 44 passed / 1 skipped (including the pyarrow selection check skipped in the Torch environment). Ruff and git diff --check passed. Remote publication/readback is recorded by the delivery response, outside this source commit.
