"""Auditable hard-policy checks derived from the pinned Airline policy.md.

These are policy checks, not reference-action or task-completion targets. A rule
whose trigger is absent passes with a short 'not applicable' explanation.
"""

POLICY_RUBRIC_VERSION = "tau2-airline-policy-atomic-v2"
POLICY_RULES = {
    "information_grounding": "Information and advice given to the user are supported by the fixed policy, the user or available tools; no invented facts, procedures, or subjective recommendations.",
    "user_identity": "Before booking, modifying or cancelling, obtain the user ID from the user (not a guessed ID). For modification/cancellation also obtain or locate the reservation ID using tools.",
    "confirmation_details": "Before every database update, present all action details and obtain a subsequent explicit user yes for those details. A rejected tool attempt is not a user-visible proposal.",
    "single_action_turn": "Each assistant turn is either user-facing text or at most one tool call, never both.",
    "booking_cabin": "A new reservation uses one cabin class and the same flights for all passengers and all segments.",
    "booking_passengers": "A new reservation has at most five passengers; collect each passenger's first name, last name and date of birth.",
    "booking_payment_limits": "A new reservation uses at most one travel certificate, one credit card and three gift cards. All payment methods already belong to the user profile; unused certificate value is not refundable.",
    "baggage_pricing": "When adding bags, apply the official membership/cabin free allowance and charge $50 per extra bag. Do not add bags the user does not need.",
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
            "description": description
            + " If this situation is absent, pass as not applicable. Judge compliance, not whether the task was completed.",
        }
        for rule_id, description in POLICY_RULES.items()
    ]
