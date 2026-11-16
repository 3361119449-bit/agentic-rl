"""Auditable checks for the policy in the frozen AReaL SFT system message.

These are policy checks, not reference-action or task-completion targets. A rule
whose trigger is absent passes with a short 'not applicable' explanation.
"""

import json

POLICY_RUBRIC_VERSION = "airline-sft-v4-policy-atomic-v6-single-tool"
POLICY_APPLICABILITY = (
    "True when the situation described by this rule occurs, including relevant "
    "requests, advice, attempts or executions; otherwise false."
)
POLICY_VERDICT_SUFFIX = (
    " If this situation is absent, return applicable=false and passed=true without "
    "violation blame. Judge compliance, not whether the task was completed."
)
POLICY_RULES = {
    "information_grounding": "Information and advice given to the user are supported by the fixed policy, the user or available tools; no invented facts, procedures, or subjective recommendations.",
    "user_identity": "Before booking, modifying or cancelling, obtain the user ID from the user (not a guessed ID). For modification/cancellation also obtain or locate the reservation ID using tools.",
    "single_action_turn": "Each assistant turn is either user-facing text or at most one tool call, never both. Independent tool calls must also be made in separate assistant turns.",
    "database_write_confirmation": "Before any action that updates the booking database, list the action details and obtain explicit user confirmation (yes) before making the write tool call.",
    "booking_cabin": "A new reservation uses one cabin class and the same flights for all passengers and all segments.",
    "booking_passengers": "A new reservation has at most five passengers; collect each passenger's first name, last name and date of birth.",
    "booking_payment_limits": "A new reservation uses at most one travel certificate, one credit card and three gift cards. All payment methods already belong to the user profile; unused certificate value is not refundable.",
    "baggage_pricing": "When adding bags, apply the supplied SFT policy's membership/cabin free allowance and charge $50 per extra bag. Do not add bags the user does not need.",
    "insurance_timing": "Insurance costs $30 per passenger and cannot be added after the initial booking.",
    "flight_change_eligibility": "Do not change the flights of a basic-economy reservation. Other flight changes must preserve origin, destination and trip type; retained segments keep their old prices. Cabin-only changes are a separate rule, not forbidden merely because the current cabin is basic economy.",
    "cabin_change_eligibility": "Do not change cabin if any reserved flight has already been flown. Otherwise cabin-only changes (including basic economy) retain flights and change all segments to the same cabin.",
    "cabin_price_difference": "For cabin changes charge the positive price difference or refund the negative difference.",
    "baggage_removal": "For an existing reservation, checked bags may be added but not removed.",
    "passenger_count": "Passenger details may be changed but the passenger count may not change, even via a human agent.",
    "modification_payment": "Flight changes require a single gift card or credit card already in the user profile for payment or refund, not a travel certificate.",
    "cancellation_reason": "Before cancellation obtain the user's cancellation reason.",
    "cancellation_eligibility": "If any portion has been flown, do not cancel and transfer is needed. Otherwise cancellation requires at least one: booking within 24 hours, airline cancellation, business cabin, or insurance covering the reason (health/weather). Basic economy is NOT a blanket cancellation prohibition.",
    "refund_destination": "Cancellation refunds go to the original payment methods within 5 to 7 business days; do not promise a different destination or schedule.",
    "compensation_request_and_facts": "Do not proactively offer compensation unless explicitly requested by the user, and confirm the facts before offering it.",
    "compensation_eligibility": "Compensation requires silver/gold membership OR travel insurance OR business cabin. Regular uninsured economy/basic economy is ineligible.",
    "compensation_reason_amount": "Only offer $100 per passenger for confirmed airline-cancelled flights, or $50 per passenger for a confirmed delay when the user wants to change/cancel AND the reservation has been changed/cancelled. No other compensation reason is allowed.",
    "transfer_scope_and_message": "Transfer if and only if the request is outside the agent's action scope; call transfer_to_human_agents first, then send 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user. Do not transfer simply to avoid policy denial.",
}


def policy_checks(task_id: str) -> list[dict[str, str]]:
    return [
        {
            "criterion_id": f"{task_id}:policy:{rule_id}",
            "applicability": POLICY_APPLICABILITY,
            "description": description + POLICY_VERDICT_SUFFIX,
        }
        for rule_id, description in POLICY_RULES.items()
    ]


def _flight_legs(flights):
    if not isinstance(flights, list) or not flights:
        return None
    if not all(
        isinstance(leg, dict)
        and isinstance(leg.get("flight_number"), str)
        and isinstance(leg.get("date"), str)
        for leg in flights
    ):
        return None
    return [(leg["flight_number"], leg["date"]) for leg in flights]


def _bag_delta(before: dict, after: dict, field: str) -> int:
    old, new = before.get(field), after.get(field)
    return new - old if type(old) is int and old >= 0 and type(new) is int else 0


def _parameter_triggered_rules(events: list[dict]) -> set[str]:
    active = set()
    reservations = {}
    reservation_tools = {
        "get_reservation_details", "book_reservation",
        "update_reservation_baggages", "update_reservation_flights",
    }
    for event in events:
        name = event.get("name")
        if name not in reservation_tools:
            continue
        args = event.get("arguments") or {}
        rid = args.get("reservation_id")
        before = reservations.get(rid, {}) if isinstance(rid, str) else {}
        if name == "book_reservation":
            if any(type(args.get(k)) is int and args[k] > 0 for k in (
                "total_baggages", "nonfree_baggages",
            )):
                active.add("baggage_pricing")
            if args.get("insurance") == "yes":
                active.add("insurance_timing")
        elif name == "update_reservation_baggages":
            snapshot = event.get("state_before") or {}
            if isinstance(rid, str) and snapshot.get("reservation_id") == rid:
                before = {**before, **snapshot}
            bag_delta = _bag_delta(before, args, "total_baggages")
            if bag_delta > 0 or _bag_delta(before, args, "nonfree_baggages") > 0:
                active.add("baggage_pricing")
            if bag_delta < 0:
                active.add("baggage_removal")
        elif name == "update_reservation_flights":
            previous = _flight_legs(before.get("flights"))
            requested = _flight_legs(args.get("flights"))
            if previous is not None and requested is not None and previous != requested:
                active.add("modification_payment")

        if not event.get("success") and event.get("db_effect") is not True:
            continue
        if name != "get_reservation_details" and isinstance(rid, str):
            # A write can invalidate the old snapshot even if its receipt is missing.
            reservations.pop(rid, None)
        if not event.get("success"):
            continue
        raw = event.get("result")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                continue
        if isinstance(raw, dict) and isinstance(raw.get("reservation_id"), str):
            received_id = raw["reservation_id"]
            if name == "book_reservation" or received_id == rid:
                reservations[received_id] = raw
    return active


def definitely_active_policy_rules(trajectory: dict, transfer_rule: dict) -> set[str]:
    """Lower bound from concrete events; absence never proves semantic N/A."""
    from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS

    events = trajectory.get("tool_events", [])
    executed = {e.get("name") for e in events if e.get("success") or e.get("db_effect") is True}
    # Tool execution errors are real attempts. Pre-call policy requirements
    # still apply; parse/schema rejections that never called the tool are not
    # evidence of execution. Outcome facts (refund/transfer) remain separate.
    attempted_events = [
        e for e in events if e.get("success") or e.get("db_effect") is True
        or e.get("error_kind") == "model_caused_execution_error"
    ]
    attempted = {e.get("name") for e in attempted_events}
    active = _parameter_triggered_rules(attempted_events)
    if events or any(m.get("role") == "assistant" for m in trajectory.get("messages", [])):
        active.update({"information_grounding", "single_action_turn"})
    if attempted & MUTATING_TOOLS:
        active.add("database_write_confirmation")
    if attempted & (MUTATING_TOOLS - {"send_certificate"}):
        active.add("user_identity")
    if "book_reservation" in attempted:
        active.update({"booking_cabin", "booking_passengers", "booking_payment_limits"})
    if "update_reservation_passengers" in attempted:
        active.add("passenger_count")
    if "cancel_reservation" in attempted:
        active.update({"cancellation_reason", "cancellation_eligibility"})
    if "cancel_reservation" in executed:
        active.add("refund_destination")
    if "send_certificate" in attempted:
        active.update({"compensation_request_and_facts", "compensation_eligibility",
                       "compensation_reason_amount"})
    if "transfer_to_human_agents" in attempted or transfer_rule.get("required"):
        active.add("transfer_scope_and_message")
    return active


def validate_policy_rows(rows: dict) -> None:
    """Reject stale official-policy rubrics before they can reach a live Judge."""
    for task_id, row in rows.items():
        if (
            row.get("policy_rubric_version") != POLICY_RUBRIC_VERSION
            or row.get("judge_checks") != policy_checks(str(task_id))
            or row.get("deterministic_rules") != []
        ):
            raise ValueError(
                f"task {task_id}: stale policy rubric; rebuild SFT-policy annotations"
            )
