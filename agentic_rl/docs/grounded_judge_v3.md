# Grounded Judge v3

The current ProCredit default uses `judge.grounding_version: judge-grounded-v3` and
`reward_version: v13-procredit-grounded-judge`. This changes Judge inputs, semantic
labels and provenance, not the reward/advantage equations or weights. The frozen
`configs/rl/airline_procredit_v4_v12.yaml` preserves the prior configuration.
Configurations without the opt-in continue to use target-v2 decisions and caches.

## Current-generation audit

The Judge returns atomic findings for amounts, authorization, query parameters,
factual grounding, policy and transfer summaries. Code derives the legacy-compatible
`bad_turn` label from confirmed findings; the response has no independent overall
bad boolean. A verified refusal cannot override an error or uncertain factual basis.
Decimal arithmetic, one-based price ranking, and specifically authorized debit caps
are checked independently against the reported operands. Operand relevance and
source interpretation still require semantic judgment; this is not a perfect oracle.

Every finding quotes the actual current generation and cites only its visible prefix.
Original actor IDs are removed from the per-turn view; target index/hash identify the
current generation. Definite wrong read-only query parameters remain errors. Mere
information collection, an ambiguous receipt acknowledgment, a hypothetical balance,
and incomplete task progress alone are not evidence of an error.

The established bound Policy occurs once in the request. Reusable context precedes
the changing target. Contradictory or invalid typed responses receive bounded feedback
retries. Unresolved responses cannot silently become neutral/positive training labels.
The original API decoding/thinking mode remains unchanged.

## Global blame and transfer reconciliation

Each failed global policy criterion requires an exact per-generation quote, an exact
Policy quote and causal evidence. Native positions, actor IDs and initial seeded
history have explicit separate meanings. Native generation boundaries are tracked
also in ordinary evaluation, independent of progress-credit calculation. Undelivered
raw output can prove an attempt, never completed execution or delivered communication.

When a neutral per-turn decision conflicts with witnessed global blame, only that
current-generation criterion receives a scoped review. A rejection removes only that
blame; unrelated confirmed violations and deterministic tool/duplicate errors remain.
Rejecting transfer-scope blame additionally requires independently reviewed, typed
transfer facts. Full existing legality, obligation, self-caused recovery, execution and
evidence validation runs again before scoring. Removing blame does not itself infer
legal execution or award completion. Invalid executed transfers cannot bypass review
by hiding the failure in an unlocalized transfer boolean.

The record stores original global output separately from the reconciled result,
primary/scoped assessments, cache keys and a composite provenance hash. Frozen label
validation rebuilds the derived decisions and reconciliation before training consumes
any advantages. New caches do not satisfy old protocols or vice versa.

## Unchanged constraints

Progress scale remains 0.5; DB/COMM/REQ and duplicate-query rules are unchanged.
Existing error <= -1 and eligible refusal >= +1 constraints remain unchanged.
RL/evaluation limits remain 35 assistant generations, with zero training signal on
truncation; training permits at most two truncation resamples and evaluation none.
Concurrency8, BF16, full tool observations and official evaluation rewards remain as
previously configured. This change does not start GPU training or new rollouts.

## Verification

Run `PYTHONPATH=src python -m pytest` and `ruff check src tests scripts`.
The grounded tests cover derived label consistency, monetary/ranking checks, exact
causal binding, legacy serialization, client retries/cache, live/frozen parity,
scoped attribution, seeded history, ordinary evaluation and transfer-facts review.
Real API validation uses separate frozen-prefix and offline80 artifacts; source
trajectory files remain unchanged. Unresolved scores are reported, never relabeled
as successful or silently replaced with zero.
