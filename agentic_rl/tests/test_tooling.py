import asyncio

import pytest

from tau2_agentic_rl.tooling import (
    ConfirmationTracker,
    execute_validated_tool_call,
    truncate_message_contents,
    validate_tool_call,
)

SCHEMA = {
    "cancel_reservation": {
        "type": "object",
        "properties": {"reservation_id": {"type": "string"}},
        "required": ["reservation_id"],
        "additionalProperties": False,
    }
}


def test_invalid_schema_never_reaches_backend() -> None:
    checked = validate_tool_call(
        "cancel_reservation",
        '{"wrong":"A"}',
        SCHEMA,
        {"cancel_reservation"},
    )
    calls = []

    async def backend(name, arguments):
        calls.append((name, arguments))

    with pytest.raises(ValueError, match="refusing backend"):
        asyncio.run(execute_validated_tool_call(checked, backend))
    assert calls == []


def test_unknown_tool_never_reaches_backend() -> None:
    checked = validate_tool_call("erase_everything", "{}", SCHEMA, set(SCHEMA))
    assert checked.error_kind == "unknown_tool"


def test_confirmation_is_argument_bound_and_one_shot() -> None:
    tracker = ConfirmationTracker()
    arguments = {"reservation_id": "A"}
    messages = [
        {
            "role": "assistant",
            "turn_idx": 1,
            "content": (
                'Please confirm. <action_proposal>{"name":"cancel_reservation",'
                '"arguments":{"reservation_id":"A"}}</action_proposal>'
            ),
        },
        {"role": "user", "turn_idx": 2, "content": "Yes, cancel it."},
    ]
    allowed, proposal = tracker.authorize("cancel_reservation", arguments, 3, messages)
    assert allowed is True
    assert proposal.confirmation_turn_id == 2
    tracker.consume(proposal.proposal_hash)
    allowed_again, _ = tracker.authorize("cancel_reservation", arguments, 4, messages)
    assert allowed_again is False


def test_changed_arguments_require_new_confirmation() -> None:
    tracker = ConfirmationTracker()
    messages = [
        {
            "role": "assistant",
            "turn_idx": 1,
            "content": (
                '<action_proposal>{"name":"cancel_reservation",'
                '"arguments":{"reservation_id":"A"}}</action_proposal>'
            ),
        },
        {"role": "user", "turn_idx": 2, "content": "Yes"},
    ]
    allowed, _ = tracker.authorize(
        "cancel_reservation", {"reservation_id": "B"}, 3, messages
    )
    assert allowed is False


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(item) for item in ids)


def test_observation_is_deterministically_truncated() -> None:
    messages, truncated = truncate_message_contents(
        [{"role": "tool", "content": "x" * 100}], TinyTokenizer(), 40
    )
    assert truncated is True
    assert len(messages[0]["content"]) <= 40
    assert messages[0]["observation_truncated"] is True
