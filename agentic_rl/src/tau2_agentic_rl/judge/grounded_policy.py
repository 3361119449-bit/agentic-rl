"""Independent per-generation witnesses for global policy blame."""

from tau2_agentic_rl.reward.simple_turn import _generation_text
from tau2_agentic_rl.versions import sha256_json

GLOBAL_VERSION = "judge-grounded-v3"


def bind_undelivered(trajectory, messages):
    from copy import deepcopy

    view = deepcopy(trajectory)
    initial = view.get("initial_prefix_length", 0)
    if type(initial) is not int or not 0 <= initial <= len(view.get("messages", [])):
        raise ValueError("invalid grounded initial history boundary")
    if any(
        m.get("role") == "assistant" and type(m.get("assistant_turn_id")) is not int
        for m in view.get("messages", [])[initial:]
    ):
        raise ValueError(
            "grounded native actors require causal annotation before binding"
        )
    known = {
        m.get("assistant_turn_id")
        for m in view.get("messages", [])
        if m.get("role") == "assistant"
    }
    extra = []
    for message in messages:
        turn = message.get("turn_idx")
        if message.get("role") != "assistant" or type(turn) is not int or turn in known:
            continue
        prior = [
            i
            for i, m in enumerate(view.get("messages", []))
            if type(m.get("assistant_turn_id")) is int and m["assistant_turn_id"] < turn
        ]
        # The first user message is visible even when no assistant was delivered.
        upper = max(prior, default=-1)
        while upper + 1 < len(view.get("messages", [])):
            following = view["messages"][upper + 1]
            actor = following.get("assistant_turn_id")
            if type(actor) is int and actor >= turn:
                break
            upper += 1
        extra.append(
            {
                "assistant_turn_id": turn,
                "generation_text": _generation_text(message),
                "causal_message_upper_bound": upper,
            }
        )
    if extra:
        view["undelivered_generations"] = extra
    return view


def actor_views(trajectory):
    views = []
    for index, message in enumerate(trajectory.get("messages", [])):
        turn = message.get("assistant_turn_id")
        if message.get("role") == "assistant" and type(turn) is int:
            text = _generation_text(message)
            views.append(
                {
                    "assistant_turn_id": turn,
                    "message_index": index,
                    "causal_message_upper_bound": index,
                    "generation_text": text,
                    "generation_sha256": sha256_json(text),
                }
            )
    known = {v["assistant_turn_id"] for v in views}
    for generation in trajectory.get("undelivered_generations", []):
        if generation["assistant_turn_id"] in known:
            raise ValueError("duplicate actor generation evidence")
        text = generation["generation_text"]
        views.append(
            {
                **generation,
                "message_index": None,
                "generation_sha256": sha256_json(text),
            }
        )
        known.add(generation["assistant_turn_id"])
    return sorted(views, key=lambda v: v["assistant_turn_id"])


def validate_global_witnesses(result, inputs, *, check_transfer_consistency=True):
    if inputs.get("judge_validation_version") != GLOBAL_VERSION:
        return
    if result.grounding_version != GLOBAL_VERSION or result.policy_attributions is None:
        raise ValueError("new global Judge requires grounded witnesses")
    views = {v["assistant_turn_id"]: v for v in actor_views(inputs["trajectory"])}
    expected = set()
    for check in result.mandatory_policy_checks:
        if check.applicable and not check.passed:
            if not check.violation_assistant_turn_ids:
                raise ValueError(
                    "failed policy criterion needs an independently localized witness"
                )
            expected.update(
                (check.criterion_id, t) for t in check.violation_assistant_turn_ids
            )
    actual = set()
    for witness in result.policy_attributions:
        key = (witness.criterion_id, witness.assistant_turn_id)
        if key in actual or key not in expected:
            raise ValueError("global witness refers to duplicate/unrequested blame")
        actual.add(key)
        view = views.get(witness.assistant_turn_id)
        if (
            not view
            or witness.quote not in view["generation_text"]
            or not witness.quote.strip()
        ):
            raise ValueError(
                "global witness quote must identify its current actor generation"
            )
        if (
            not witness.policy_quote.strip()
            or witness.policy_quote not in inputs["policy"]
        ):
            raise ValueError("global witness has no exact bound policy quote")
        ids = witness.evidence_message_indices
        if (
            len(ids) != len(set(ids))
            or (view["message_index"] is not None and view["message_index"] not in ids)
            or any(i < 0 or i > view["causal_message_upper_bound"] for i in ids)
        ):
            raise ValueError("global witness must cite its target within causal prefix")
    if actual != expected:
        raise ValueError("global witness coverage differs from failed policy blame")
    executed = any(
        e.get("name") == "transfer_to_human_agents" and e.get("success")
        for e in inputs["trajectory"].get("tool_events", [])
    )
    scope_failed = any(
        c.applicable and not c.passed and c.violation_assistant_turn_ids
        for c in result.mandatory_policy_checks
        if c.criterion_id.endswith(":policy:transfer_scope_and_message")
    )
    if (
        check_transfer_consistency
        and executed
        and not result.transfer_check.valid
        and not scope_failed
    ):
        raise ValueError("invalid transfer requires localized witnessed scope failure")


def policy_candidates(result, assistant_turn_id):
    return [
        x.model_dump()
        for x in (result.policy_attributions or [])
        if x.assistant_turn_id == assistant_turn_id
    ]


def reconcile_witnesses(result, assistant_turn_id, reviews, inputs):
    validate_global_witnesses(result, inputs, check_transfer_consistency=False)
    candidates = policy_candidates(result, assistant_turn_id)
    if len(reviews) != len(candidates) or {x["criterion_id"] for x in reviews} != {
        x["criterion_id"] for x in candidates
    }:
        raise ValueError("policy conflict review coverage mismatch")
    revised = result.model_copy(deep=True)
    changed = set()
    by_id = {x["criterion_id"]: x for x in reviews}
    for review in reviews:
        if review["verdict"] == "uncertain":
            raise ValueError("unresolved global policy review")
        if review["verdict"] not in {"confirmed", "rejected"}:
            raise ValueError("invalid review verdict")
        if review["verdict"] == "rejected":
            changed.add(review["criterion_id"])
    revised.policy_attributions = [
        x
        for x in revised.policy_attributions
        if not (x.assistant_turn_id == assistant_turn_id and x.criterion_id in changed)
    ]
    for check in revised.mandatory_policy_checks:
        if check.criterion_id not in by_id:
            continue
        if check.criterion_id in changed:
            check.violation_assistant_turn_ids = [
                t for t in check.violation_assistant_turn_ids if t != assistant_turn_id
            ]
            check.passed = not bool(check.violation_assistant_turn_ids)
            remaining = [
                x.reason
                for x in revised.policy_attributions
                if x.criterion_id == check.criterion_id
            ]
            check.short_reason = (
                " ".join(remaining)
                if remaining
                else "Scoped causal review rejected prior blame: "
                + by_id[check.criterion_id]["reason"]
            )
    # Full transfer consistency is checked after a separately reviewed facts result;
    # never infer valid/required/self_caused_only from deleting policy blame.
    return revised


def grounded_rubric(trajectory):
    return {
        "grounding_version": GLOBAL_VERSION,
        "actor_generation_views": [
            {
                k: v
                for k, v in x.items()
                if k != "generation_text" or x["message_index"] is None
            }
            for x in actor_views(trajectory)
        ],
        "rules": (
            "Reference expected actions/totals measure task completion, not factual truth or policy legality. "
            "A correct partial refund subtotal is not fabricated because other task goals remain incomplete. "
            "Do not require the literal word yes when the latest user unambiguously authorizes listed details. "
            "Do not inherit earlier mistakes onto a later corrected action matching the latest authorization. "
            "Basic-economy flight change restriction does not ban same-flight cabin change followed by a "
            "separately confirmed flight change. Economy may switch connection/direct preserving endpoints/type. "
            "Independently check all prices, preserved-leg original fares, passenger count, signed refunds/charges "
            "and latest monetary authorization. Transfer legality never establishes summary factual accuracy. "
            "Every failed policy check needs one policy_attributions item for EACH offending actor ID. Copy "
            "assistant_turn_id, exact quote from that actor_generation_view.generation_text, exact policy_quote "
            "from FIXED_AIRLINE_POLICY, causal evidence_message_indices including view.message_index and no "
            "larger index, plus concise reason. These indices are zero-based POSITIONS in trajectory.messages, "
            "NOT messages.turn_idx or actor IDs. Undelivered generations have message_index=null; "
            "cite only native positions up to causal_message_upper_bound, and treat this raw output as an "
            "attempt, never delivered communication or a completed write. A current write's later tool result cannot prove what was known "
            "before the write: derive foreseeable amount from prior facts instead. Passed/inapplicable checks "
            "have no witnesses. Avoid all future facts or hidden task answers as proof of current policy blame. "
            "For ambiguous found after a supplied booking ID, hypothetical would-cover balance, incomplete "
            "task alone, or harmless information collection, do not manufacture a policy violation. "
            "If an executed transfer is judged invalid, an applicable failed transfer_scope_and_message "
            "criterion and current-generation witnesses are REQUIRED; valid=false may not bypass attribution. "
            "Other existing mandatory and transfer rules retain their exact meaning."
        ),
        "witness_schema": {
            "criterion_id": "requested failed policy ID",
            "assistant_turn_id": 1,
            "quote": "exact current generation substring",
            "policy_quote": "exact bound policy substring",
            "evidence_message_indices": [0],
            "reason": "independent current-generation evidence",
        },
    }
