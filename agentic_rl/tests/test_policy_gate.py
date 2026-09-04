from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.schemas import JudgeResult, OfficialScores, ToolEvent


def test_unconfirmed_write_hard_gates_reward() -> None:
    action = {
        "action_id": "a",
        "name": "cancel_reservation",
        "arguments": {"reservation_id": "A"},
    }
    event = ToolEvent(
        event_id="e",
        sequence=0,
        turn_id=1,
        name=action["name"],
        arguments=action["arguments"],
        success=True,
        db_effect=True,
        confirmed_before=False,
    )
    result = score_trajectory(
        events=[event],
        messages=[],
        assistant_turns=1,
        required_actions=[action],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
    )
    assert result.policy_gate is False
    assert result.train_reward == 0.0


def test_multiple_calls_in_one_turn_hard_gate_reward() -> None:
    events = [
        ToolEvent(
            event_id=f"e{i}", sequence=i, turn_id=1, name="search_flights", success=True
        )
        for i in range(2)
    ]
    result = score_trajectory(
        events=events,
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
    )
    assert result.train_reward == 0.0


def test_unannotated_write_is_a_separate_task_safety_gate() -> None:
    event = ToolEvent(
        event_id="e",
        sequence=0,
        turn_id=1,
        name="cancel_reservation",
        arguments={"reservation_id": "WRONG"},
        success=True,
        db_effect=True,
        confirmed_before=True,
    )
    result = score_trajectory(
        events=[event],
        messages=[],
        assistant_turns=1,
        required_actions=[
            {
                "action_id": "a",
                "name": "cancel_reservation",
                "arguments": {"reservation_id": "RIGHT"},
            }
        ],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
    )
    assert result.policy_gate is True
    assert result.task_safety_gate is False
    assert result.train_reward == 0
