# Judge Lite Implementation Plan

> **For agentic workers:** use superpowers:executing-plans; implement inline with local tests only.

**Goal:** improve Judge guidance with the existing call structure and compact outputs.
**Architecture:** retain target-v2 flat labels and original actor/retry paths. Edit two prompts; add narrow response-consistency validation inside existing transport validation and useful feedback in its existing retry loop.
**Tech Stack:** Python/Pydantic/httpx/pytest/ruff.
**Spec:** task workspace SPEC_LITE.md; latest user forbids online scoring.

## Constraints
No new LLM stage, output schema, reward equation, weight, protocol hash or decoding change. No scoring API/GPU/rollout. Preserve discarded experiment in archive branch; current draft supersedes it.

## Task1: compact guidance and local validation
- [ ] Add failing tests for contradiction versus neutral/quoted/negated reasons, current bad/refusal conflict, bounded feedback and cache reuse.
- [ ] Implement concise current-turn prompt, small local consistency check, global blame/fact guidance and existing-loop feedback.
- [ ] Run focused and complete tests with socket guard, lint and preserved-formula checks.

## Task2: sync and verify
- [ ] Fresh static review, fix material issues locally, no scoring APIs.
- [ ] Replace draft PR9 content with minimal diff; run offline GitHub CI.
- [ ] Server downloads GitHub ZIP directly; preserve local overlays, verify and sync selected files after checks.
- [ ] Confirm zero online experiment runners and unchanged data/model inventories; report validation limits.
