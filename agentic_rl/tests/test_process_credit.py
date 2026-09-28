import pytest
from test_policy_credit import policy_row

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.schemas import ToolEvent


def process_rows(cost=0.1):
    result = []
    for i in range(8):
        row = policy_row(i, valid=True, phi=[0, 0, 0, 0])
        row["policy_credit"].update(violating_turns=[])
        row["process_credit"] = {"version": "process-credit-v1", "turn_costs": [0, cost, 0]}
        result.append(row)
    return result


def test_zero_score_group_retains_local_process_cost():
    result = compute_group_credit(process_rows(), CreditConfig(version="procredit-turn-v3"))
    assert result["has_signal"]
    assert result["trajectories"][0]["token_advantages"] == [0, 0, 0, -0.1, 0]
    assert not compute_group_credit(process_rows(), CreditConfig(
        version="procredit-turn-v3", turn_coefficient=0))["has_signal"]


def test_policy_and_process_penalty_on_same_turn_are_not_added_twice():
    rows = process_rows()
    for row in rows:
        row["valid"] = False
        row["policy_credit"]["violating_turns"] = [1]
    result = compute_group_credit(rows, CreditConfig(version="procredit-turn-v3"))
    assert result["trajectories"][0]["turn_advantages"] == [0, -1, 0]


def test_invalid_soft_turn_limit_does_not_attribute_cost_to_negative_index():
    from tau2_agentic_rl.reward.process_penalty import (
        ProcessPenaltyConfig,
        build_process_credit,
    )

    with pytest.raises(ValueError, match="turn"):
        build_process_credit([], 2, ProcessPenaltyConfig(soft_turn_limit=-1))


def test_local_process_trace_has_same_cap_and_overlimit_turns_as_scalar():
    from tau2_agentic_rl.reward.process_penalty import (
        ProcessPenaltyConfig,
        build_process_credit,
    )

    events = [ToolEvent(event_id=f"e{i}", sequence=i, turn_id=i+1, error_kind="parse_error")
              for i in range(3)]
    trace = build_process_credit(events, 18, ProcessPenaltyConfig())
    assert sum(trace["turn_costs"]) == pytest.approx(.2)
    assert len(trace["turn_costs"]) == 18
    assert trace["turn_costs"][3:15] == [0]*12
    assert all(x > 0 for x in trace["turn_costs"][15:])
    with pytest.raises(ValueError, match="duplicate"):
        build_process_credit(events + [events[0]], 18, ProcessPenaltyConfig())


def test_v2_cannot_silently_discard_v3_process_cost():
    with pytest.raises(ValueError, match="process"):
        compute_group_credit(process_rows(), CreditConfig(version="procredit-turn-v2"))
