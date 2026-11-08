"""Grounded current-turn Judge protocol with bounded, cached consistency review."""

from dataclasses import replace

from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.judge.simple_turn import PROMPT as LEGACY_PROMPT
from tau2_agentic_rl.judge.simple_turn import build_messages as legacy_messages
from tau2_agentic_rl.reward.grounded import (
    GROUNDED_VERSION,
    GroundedAssessment,
    validate_assessment,
)

PROMPT = (
    LEGACY_PROMPT.split("bad_turn is true")[0]
    + """
Audit the ENTIRE current generation in six categories: amount, authorization,
parameters, grounding, policy, transfer_summary. Do not let a correct subsection
excuse a wrong subsection. No continuous quality score and no separate bad_turn
boolean: the caller derives it from confirmed errors and verified amount checks.
For every applicable concern give a SHORT atomic check: category, verdict
(ok/error/uncertain), an exact quote from current target, causal evidence indices,
and one concise reason. An uncertain fact is not proof of its opposite.
Neutral generations may have checks=[]; audited_categories must list all six.
Do not output internal deliberation or a second overall verdict in neutral_reason.

PARAMETERS: recover the LATEST explicit user target and allowed flexibility from
user/tool facts BEFORE this generation. A read-only search is still wrong if its
booking/date/origin/destination clearly conflicts with that known target. Do not
invent a blanket read-only exemption. User-authorized alternative dates, relevant
comparison/historical searches, and ordinary empty results are NOT automatically
wrong. Earlier assistant inventions do not amend user instructions. Never confuse
hours of flexibility with days, or interchange endpoints/direction. Flights between
the two user-specified origin airports are not their requested outbound/return to
a different region. Unknown target parameters alone do not establish an error.

MONEY: when quoting, reporting settlement, ranking fares, or executing a potentially
charging write, independently check price sources, passenger count, existing paid
fare versus new total versus DIFFERENCE, preserved unchanged-leg ORIGINAL price,
insurance/baggage effects, and signed charge/refund direction. Never reprice an
unchanged leg when policy preserves its stored fare. Do not use a historical ticket
price as proof of current new-booking price. Use latest tool/user facts, not earlier
assistant arithmetic. No charge exceeding the latest specifically authorized amount
is authorized merely because the user agreed to cabin/flights/payment ID. Foreseeable
write costs use prior visible fares/count, NOT a later tool result hidden from this
prefix. A successful tool call is not policy or authorization evidence by itself.
Fill amount_checks for applicable definite amounts using sourced operands and a
simple numeric + - * / expression, expected exact result, and the target reported
amount if any. For a charging write use kind=authorization and authorized=user's
specific charge limit. Negative net refunds do not exceed a debit limit. For ranking
list ALL relevant available candidate totals, one-based rank, expected ranked price
and reported target price; include direct and connecting options when user asked
for overall ranking. Explain source/direction/units briefly in reason. The caller
checks the supplied arithmetic/ranking; do not invent operands. If essential price
facts are absent, use an uncertain check instead of pretending to know the price.

FACTS: status:null is not evidence that airline did not cancel. Checking one of two
flights does not verify both. Tool capabilities do not include imaginary notes,
email, seat-selection or saved-profile writes. Separate attempt, hypothetical plan,
and successful execution; 'would cover X, leaving Y' is not by itself a claim of a
successful debit. 'I found your reservation' after user supplied its ID is ambiguous;
do not manufacture a factual fabrication unless it asserts unsupported verified
attributes or an actual completed lookup. Merely asking for date/time or other missing details is not a promise or execution of a prohibited change; do not infer illegal advice from harmless information collection alone. A failed write cannot justify a later
claim of success or account balance. Transfer summaries must independently get
money, date arithmetic, user requests and cause right, even if transfer is legal.
Correct partial totals are not false merely because an incomplete full task has a
different expected total. Incomplete task alone is NOT bad. Do not invent policy
limits on dates, airports/cabins, insurance, or new bookings. Respect Policy
exceptions: a current basic-economy flight-change restriction is distinct from a
permitted same-flight cabin change followed by a newly authorized flight change.
User confirmation can be an unequivocal instruction after listed details, not only
the literal word 'yes'. Earlier mistakes cannot contaminate a later corrected action
matching the latest authorization. Do not treat human transfer as a policy waiver.

REFUSAL: refusal_verified=true ONLY for a clearly requested prohibited operation
with sufficient visible facts, accurate complete explanation and no bad part.
Unverified definite factual claims defeat a clean refusal even if denial direction
might be right. Add refusal_requests with exact {target,operation}; reuse previous
request keys. Asking missing facts, incomplete help, vague statements or task failure
are not refusal awards. Preserve prior refusal revocation rules: only the SAME
still-illegal request now explicitly committed/executed can revoke it. Different
requests, new facts, or unrelated transfer do not revoke a valid earlier refusal.

GLOBAL REVIEW: policy_candidates, if supplied, are disputed prior Judge hypotheses,
NOT authority. For each return policy_reviews with criterion_id, verdict
(confirmed/rejected/uncertain), exact target quote, exact bound policy_quote,
causal evidence_message_indices and concise reason. Independently check the actual
current action, latest user authorization and all applicable Policy exceptions.
A rejection must explain why THIS proposed blame is unsupported. An uncertain
review cannot be accepted as neutral; it will be retried. Do not use future evidence.

Return JSON only; no extra fields. Required structure:
{"turn_index":0,"target_message_index":0,"target_generation_sha256":"copy request hash",
"target_quote":"exact target substring","audited_categories":["amount","authorization",
"parameters","grounding","policy","transfer_summary"],"checks":[],"amount_checks":[],
"refusal_verified":false,"refusal_requests":[],"revoked_refusals":[],"policy_reviews":[],
"neutral_reason":"one brief neutral explanation; no independent overall verdict"}
Each checks item: {"category":"grounding","verdict":"error","quote":"exact current target quote",
"evidence_message_indices":[0],"reason":"short supported finding"}.
Each amount_checks item: {"kind":"arithmetic","expression":"100-20","expected":80,
"reported":75,"authorized":null,"candidates":null,"rank":null,"quote":"exact current target quote",
"evidence_message_indices":[0],"reason":"source and units"}.
kind is arithmetic/authorization/ranking. For ranking expression=null and provide
candidates/rank; for authorization reported may be null, authorized must be present.
Each policy_reviews item: {"criterion_id":"copy candidate ID","verdict":"rejected",
"quote":"exact target substring","policy_quote":"exact supplied policy substring",
"evidence_message_indices":[0],"reason":"current-target independent rationale"}.
refusal_requests items are {"target":"object","operation":"prohibited operation"}.
revoked_refusals items additionally include earlier zero-based turn_index.
"""
)


def build_messages(**request):
    messages = legacy_messages(**request)
    messages[0]["content"] = PROMPT
    return messages


class GroundedTurnJudge(DeepSeekJudge):
    result_model = GroundedAssessment
    build_messages = staticmethod(build_messages)

    @staticmethod
    def _validate_requested_criteria(result, inputs):
        validate_assessment(result.model_dump(), inputs)


def client_from(judge):
    return GroundedTurnJudge(
        replace(
            judge.config,
            cache_dir=str(judge.cache_dir / "simple_turn_target_v3"),
            prompt_version=GROUNDED_VERSION,
            rubric_version=GROUNDED_VERSION,
            schema_version=GROUNDED_VERSION,
            scorer_code_version=GROUNDED_VERSION,
        )
    )


def _review_request(request, candidates):
    from copy import deepcopy

    view = deepcopy(request)
    # These are hypotheses, not another set of actor/message number selectors.
    view["policy_candidates"] = [
        {
            k: v
            for k, v in candidate.items()
            if k in ("criterion_id", "quote", "policy_quote", "reason")
        }
        for candidate in candidates
    ]
    return view


async def prepare_grounded_turns(
    inputs, global_inputs, result, judge, *, progress=None
):
    """Resolve current/global semantic conflicts BEFORE unchanged reward scoring."""
    from tau2_agentic_rl.judge.grounded_policy import (
        policy_candidates,
        reconcile_witnesses,
        validate_global_witnesses,
    )
    from tau2_agentic_rl.reward.simple_turn import turn_request
    from tau2_agentic_rl.versions import sha256_json

    validate_global_witnesses(result, global_inputs)
    client = client_from(judge)
    original = result.model_dump()
    decisions, assessments, calls = [], [], []
    for t in range(len(inputs["turn_message_indices"])):
        request = turn_request(inputs, t, decisions)
        model, raw, prompt_hash, cache_key = await client.evaluate(**request)
        primary, decision = validate_assessment(model.model_dump(), request)
        entry = {
            "primary": primary,
            "primary_request_sha256": sha256_json(request),
            "review": None,
        }
        calls.append(
            {
                "turn_index": t,
                "phase": "primary",
                "raw": raw,
                "prompt_hash": prompt_hash,
                "cache_key": cache_key,
            }
        )
        candidates = policy_candidates(result, t + 1)
        if candidates and not decision["bad_turn"]:
            review_request = _review_request(request, candidates)
            model, raw, prompt_hash, cache_key = await client.evaluate(**review_request)
            review, decision = validate_assessment(model.model_dump(), review_request)
            result = reconcile_witnesses(
                result, t + 1, review["policy_reviews"], global_inputs
            )
            entry.update(
                review=review, review_request_sha256=sha256_json(review_request)
            )
            calls.append(
                {
                    "turn_index": t,
                    "phase": "policy_conflict_review",
                    "raw": raw,
                    "prompt_hash": prompt_hash,
                    "cache_key": cache_key,
                }
            )
        assessments.append(entry)
        decisions.append(decision)
        if progress is not None:
            progress()
    transfer_audit = None
    scope_reviews = _scope_corrections(assessments)
    if scope_reviews:
        from tau2_agentic_rl.judge.grounded_transfer import (
            apply_transfer_review,
            transfer_request,
        )
        from tau2_agentic_rl.judge.grounded_transfer import (
            client_from as transfer_client,
        )

        view = transfer_request(global_inputs, result, scope_reviews)
        facts, raw, prompt_hash, cache_key = await transfer_client(judge).evaluate(
            **view
        )
        result = apply_transfer_review(result, facts, global_inputs)
        transfer_audit = {
            "assessment": facts.model_dump(),
            "request_sha256": sha256_json(view),
            "raw": raw,
            "prompt_hash": prompt_hash,
            "cache_key": cache_key,
        }
        if progress is not None:
            progress()
    validate_global_witnesses(result, global_inputs)
    prepared = {
        "version": GROUNDED_VERSION,
        "transfer_review": transfer_audit,
        "global_original": original,
        "global_reconciled_sha256": sha256_json(result.model_dump()),
        "assessments": assessments,
        "decisions": decisions,
        "judge_calls": calls,
    }
    validate_prepared(inputs, global_inputs, result, prepared)
    return result, prepared


def validate_prepared(inputs, global_inputs, result, prepared):
    """Rebuild every derived label/reconciliation from the frozen typed findings."""
    from tau2_agentic_rl.judge.grounded_policy import (
        policy_candidates,
        reconcile_witnesses,
        validate_global_witnesses,
    )
    from tau2_agentic_rl.reward.simple_turn import turn_request
    from tau2_agentic_rl.schemas import JudgeResult
    from tau2_agentic_rl.versions import sha256_json

    if prepared.get("version") != GROUNDED_VERSION:
        raise ValueError("missing grounded audit version")
    original = JudgeResult.model_validate(prepared["global_original"])
    validate_global_witnesses(original, global_inputs)
    entries = prepared["assessments"]
    if len(entries) != len(inputs["turn_message_indices"]):
        raise ValueError("grounded assessment coverage changed")
    decisions = []
    for t, entry in enumerate(entries):
        request = turn_request(inputs, t, decisions)
        if entry.get("primary_request_sha256") != sha256_json(request):
            raise ValueError("primary target changed")
        _, decision = validate_assessment(entry["primary"], request)
        candidates = policy_candidates(original, t + 1)
        if candidates and not decision["bad_turn"]:
            view = _review_request(request, candidates)
            if entry.get("review_request_sha256") != sha256_json(view):
                raise ValueError("review request changed")
            review, decision = validate_assessment(entry.get("review"), view)
            original = reconcile_witnesses(
                original, t + 1, review["policy_reviews"], global_inputs
            )
        elif entry.get("review") is not None:
            raise ValueError("unexpected policy conflict review")
        decisions.append(decision)
    scope_reviews = _scope_corrections(entries)
    transfer_audit = prepared.get("transfer_review")
    if scope_reviews:
        from tau2_agentic_rl.judge.grounded_transfer import (
            TransferFactsResult,
            apply_transfer_review,
            transfer_request,
        )

        view = transfer_request(global_inputs, original, scope_reviews)
        if not transfer_audit or transfer_audit.get("request_sha256") != sha256_json(
            view
        ):
            raise ValueError("missing or changed independent transfer review")
        original = apply_transfer_review(
            original,
            TransferFactsResult.model_validate(transfer_audit["assessment"]),
            global_inputs,
        )
    elif transfer_audit is not None:
        raise ValueError("unexpected transfer review")
    validate_global_witnesses(original, global_inputs)
    if (
        prepared.get("decisions") != decisions
        or original.model_dump() != result.model_dump()
        or prepared.get("global_reconciled_sha256") != sha256_json(result.model_dump())
    ):
        raise ValueError(
            "grounded decisions or global reconciliation differ from evidence"
        )
    return decisions


def finish_grounded_trace(inputs, prepared, events, policy_credit):
    from tau2_agentic_rl.reward.simple_turn import build_label_trace

    trace = build_label_trace(inputs, prepared["decisions"], events, policy_credit)
    trace["grounded_audit"] = prepared
    trace["judge_calls"] = prepared["judge_calls"]
    return trace


def _scope_corrections(entries):
    return [
        {"assistant_turn_id": t + 1, **review}
        for t, entry in enumerate(entries)
        for review in (entry.get("review") or {}).get("policy_reviews", [])
        if review["verdict"] == "rejected"
        and review["criterion_id"].endswith(":policy:transfer_scope_and_message")
    ]


def result_provenance(prepared, cache_key):
    from tau2_agentic_rl.versions import sha256_json

    return {
        "version": "grounded-composite-v3",
        "global_cache_key": cache_key,
        "global_original_sha256": sha256_json(prepared["global_original"]),
        "global_reconciled_sha256": prepared["global_reconciled_sha256"],
        "grounded_audit_sha256": sha256_json(prepared),
    }
