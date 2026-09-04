from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.reward.transfer_branch import TRANSFER_MESSAGE
from tau2_agentic_rl.schemas import (
    JudgeResult,
    OfficialScores,
    ToolEvent,
    TransferCheck,
)


def transfer_event() -> ToolEvent:
    return ToolEvent(
        event_id="t",
        sequence=0,
        turn_id=1,
        name="transfer_to_human_agents",
        arguments={"summary": "cannot change destination"},
        success=True,
    )


def test_valid_required_transfer_can_score_one() -> None:
    result = score_trajectory(
        events=[transfer_event()],
        messages=[{"role": "assistant", "content": TRANSFER_MESSAGE}],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1),
        judge=JudgeResult(transfer_check=TransferCheck(applicable=True, valid=True)),
        transfer_rule={
            "allowed": True,
            "required": True,
            "required_communication_checks": [TRANSFER_MESSAGE],
            "required_pre_transfer_action_groups": [],
            "semantic_checks": [],
        },
    )
    assert result.branch == "human_transfer"
    assert result.train_reward == 1.0


def test_illegal_transfer_is_zero() -> None:
    result = score_trajectory(
        events=[transfer_event()],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=0),
        judge=JudgeResult(),
        transfer_rule={"allowed": False},
    )
    assert result.train_reward == 0.0


def test_missing_required_transfer_is_zero() -> None:
    result = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
        transfer_rule={"allowed": True, "required": True},
    )
    assert result.branch == "normal"
    assert result.train_reward == 0.0


def test_transfer_semantic_uses_its_own_judge_items() -> None:
    result = score_trajectory(
        events=[transfer_event()],
        messages=[{"role": "assistant", "content": TRANSFER_MESSAGE}],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1),
        judge=JudgeResult(
            transfer_semantic_checks=[
                {
                    "criterion_id": "transfer-valid",
                    "passed": True,
                }
            ],
            transfer_check=TransferCheck(applicable=True, valid=True),
        ),
        transfer_rule={
            "allowed": True,
            "semantic_checks": [{"criterion_id": "transfer-valid"}],
        },
    )
    assert result.components["transfer_semantic"].value == 1.0
