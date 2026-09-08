from typing import get_args

import pytest

from tau2_agentic_rl.reward.process_penalty import (
    ProcessPenaltyConfig,
    compute_process_penalty,
)
from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.schemas import (
    JudgeResult,
    OfficialScores,
    ToolErrorKind,
    ToolEvent,
)
from tau2_agentic_rl.tooling import validate_tool_turn


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


@pytest.mark.parametrize("kind", get_args(ToolErrorKind))
def test_every_valid_tool_error_has_a_base_penalty(kind):
    event = ToolEvent(event_id="e", sequence=0, turn_id=0, error_kind=kind)
    expected = ProcessPenaltyConfig().penalties[kind]
    result = compute_process_penalty([event], assistant_turns=1)
    assert result.penalty == pytest.approx(expected)
    assert result.events == [{"event_id": "e", "kind": kind, "penalty": expected}]


def test_mixed_text_and_tool_call_reaches_final_reward_without_scoring_failure():
    text = (
        'I will check. <tool_call>{"name":"get_reservation_details",'
        '"arguments":{"reservation_id":"ABC123"}}</tool_call>'
    )
    kind = validate_tool_turn(text, 1)
    assert kind == "mixed_content_and_tool_call"
    event = ToolEvent(event_id="e", sequence=0, turn_id=0, error_kind=kind)
    result = score_trajectory(
        events=[event],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
    )
    assert result.process_penalty == pytest.approx(0.1)
    assert result.train_reward == pytest.approx(0.9)
