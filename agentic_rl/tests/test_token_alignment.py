import pytest

from tau2_agentic_rl.schemas import TokenTurn
from tau2_agentic_rl.token_alignment import (
    validate_aligned_response,
    validate_pre_update_ratio,
)


def test_policy_tokens_and_old_log_probs_stay_aligned() -> None:
    turns = [
        TokenTurn(
            assistant_turn_index=0,
            prompt_token_ids=[1],
            output_token_ids=[2, 3],
            output_old_log_probs=[-0.1, -0.2],
        ),
        TokenTurn(
            assistant_turn_index=1,
            prompt_token_ids=[1, 2, 3, 4],
            output_token_ids=[5],
            output_old_log_probs=[-0.3],
        ),
    ]
    validate_aligned_response(
        response_ids=[2, 3, 4, 5],
        response_mask=[1, 1, 0, 1],
        aligned_old_log_probs=[-0.1, -0.2, 0.0, -0.3],
        raw_turns=turns,
    )


def test_non_policy_log_prob_must_be_zero() -> None:
    with pytest.raises(ValueError, match="non-policy"):
        validate_aligned_response([1], [0], [-1.0], [])


def test_pre_update_ratio_guard() -> None:
    validate_pre_update_ratio([-0.101], [-0.1], [1], tolerance=0.01)
    with pytest.raises(ValueError, match="misaligned"):
        validate_pre_update_ratio([-0.2], [-0.1], [1], tolerance=0.01)
