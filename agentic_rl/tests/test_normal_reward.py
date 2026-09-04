import pytest

from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, OfficialScores


def test_partial_components_are_weighted_and_inapplicable_items_omitted() -> None:
    result = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(
            reward=0.0,
            db_applicable=True,
            db_score=1.0,
            communicate_applicable=True,
            communicate_partial=0.5,
        ),
        judge=JudgeResult(semantic_checks=[JudgeCheck(criterion_id="s", passed=False)]),
    )
    # (0.30*1 + 0.15*0.5 + 0.25*0) / 0.70, then 0.75 progress.
    assert result.progress == pytest.approx(0.375 / 0.70)
    assert result.train_reward == pytest.approx(0.75 * 0.375 / 0.70)
    assert result.strict_success == 0.0


def test_all_applicable_components_pass_gives_one() -> None:
    result = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
    )
    assert result.train_reward == 1.0
    assert result.strict_success == 1.0
