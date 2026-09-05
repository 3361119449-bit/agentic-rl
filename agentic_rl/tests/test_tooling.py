import asyncio

import pytest

from tau2_agentic_rl.tooling import (
    ConfirmationTracker,
    execute_validated_tool_call,
    truncate_message_contents,
    validate_tool_call,
    validate_tool_turn,
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
    tracker.observe_visible_assistant_text(messages[0]["content"], 1)
    tracker.observe_user_reply(messages[1]["content"], 2)
    allowed, proposal = tracker.authorize("cancel_reservation", arguments)
    assert allowed is True
    assert proposal.confirmation_turn_id == 2
    tracker.consume(proposal.proposal_hash)
    allowed_again, _ = tracker.authorize("cancel_reservation", arguments)
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
    tracker.observe_visible_assistant_text(messages[0]["content"], 1)
    tracker.observe_user_reply(messages[1]["content"], 2)
    allowed, _ = tracker.authorize("cancel_reservation", {"reservation_id": "B"})
    assert allowed is False


def test_blocked_write_does_not_establish_invisible_proposal() -> None:
    tracker = ConfirmationTracker()
    args = {"reservation_id": "A"}
    assert tracker.authorize("cancel_reservation", args) == (False, None)
    assert tracker.pending is None
    tracker.observe_visible_assistant_text("Please confirm", 1)
    tracker.observe_user_reply("Yes", 2)
    assert tracker.authorize("cancel_reservation", args) == (False, None)


def test_proposal_with_tool_call_cannot_authorize_write() -> None:
    tracker = ConfirmationTracker()
    tracker.observe_visible_assistant_text(
        '<action_proposal>{"name":"cancel_reservation","arguments":{"reservation_id":"A"}}</action_proposal>'
        '<tool_call>{"name":"cancel_reservation","arguments":{"reservation_id":"A"}}</tool_call>',
        1,
    )
    tracker.observe_user_reply("Yes", 2)
    assert tracker.authorize("cancel_reservation", {"reservation_id": "A"}) == (
        False,
        None,
    )


def test_reply_must_follow_proposal_and_can_revoke_confirmation() -> None:
    tracker = ConfirmationTracker()
    tracker.observe_user_reply("Yes", 0)
    tracker.observe_visible_assistant_text(
        '<action_proposal>{"name":"cancel_reservation","arguments":{"reservation_id":"A"}}</action_proposal>',
        1,
    )
    assert tracker.authorize("cancel_reservation", {"reservation_id": "A"}) == (
        False,
        None,
    )
    tracker.observe_user_reply("Yes", 2)
    tracker.observe_user_reply("No, wait", 3)
    assert tracker.authorize("cancel_reservation", {"reservation_id": "A"}) == (
        False,
        None,
    )


CALL = '<tool_call>{"name":"x","arguments":{}}</tool_call>'


@pytest.mark.parametrize(
    "reply,allowed",
    [
        ("Yes", True),
        ("Yes, please proceed.", True),
        ("Yes, but cancel B instead", False),
        ("Yes, cancel B", False),
        ("Yes, only if the fee is zero", False),
    ],
)
def test_confirmation_cannot_hide_changed_or_conditional_instructions(reply, allowed):
    tracker = ConfirmationTracker()
    tracker.observe_visible_assistant_text(
        '<action_proposal>{"name":"cancel_reservation","arguments":{"reservation_id":"A"}}</action_proposal>',
        1,
    )
    tracker.observe_user_reply(reply, 2)
    assert (
        tracker.authorize("cancel_reservation", {"reservation_id": "A"})[0] is allowed
    )


@pytest.mark.parametrize(
    "raw,count,error",
    [
        (CALL + "<|im_end|>", 1, None),
        ("Sure. " + CALL, 1, "mixed_content_and_tool_call"),
        (CALL + "<tool_call>{broken}</tool_call>", 1, "parse_error"),
        (CALL + "<tool_call>", 1, "parse_error"),
        (CALL * 2, 2, "multiple_tool_calls"),
        ("<|im_end|>arbitrary" + CALL, 1, "mixed_content_and_tool_call"),
        (
            "<action_proposal>{}</action_proposal>" + CALL,
            1,
            "mixed_content_and_tool_call",
        ),
        ("Hello", 0, None),
    ],
)
def test_whole_tool_turn_is_validated(raw, count, error):
    assert validate_tool_turn(raw, count) == error


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
