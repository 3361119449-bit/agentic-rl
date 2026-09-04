import pytest

from tau2_agentic_rl.reward.process_penalty import compute_process_penalty
from tau2_agentic_rl.schemas import ToolEvent


def test_one_base_error_plus_independent_retry_and_turn_penalties() -> None:
    events = [
        ToolEvent(
            event_id="e0",
            sequence=0,
            turn_id=1,
            name="bad",
            error_kind="unknown_tool",
            unchanged_retry=True,
        ),
        ToolEvent(
            event_id="e1",
            sequence=1,
            turn_id=2,
            name="read",
            no_progress=True,
        ),
    ]
    result = compute_process_penalty(events, assistant_turns=17)
    assert result.penalty == pytest.approx(0.20)
    assert [item["kind"] for item in result.events] == [
        "unknown_tool",
        "unchanged_retry",
        "duplicate_no_progress",
        "over_soft_turn_limit",
    ]


def test_reasonable_failed_tool_without_model_error_is_not_penalized() -> None:
    event = ToolEvent(event_id="e", sequence=0, turn_id=1, name="search", success=False)
    assert compute_process_penalty([event], 1).penalty == 0.0
