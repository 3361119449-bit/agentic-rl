from copy import deepcopy

import pytest

from tau2_agentic_rl.reward import progress
from tau2_agentic_rl.schemas import ToolEvent


def build(**changes):
    args = dict(
        task={"id": "0", "evaluation_criteria": {"communicate_info": ["refund"]}},
        messages=[
            {"role": "user", "content": "refund"},
            {"role": "assistant", "content": "refund"},
            {"role": "assistant", "content": "done"},
        ],
        prefix_lengths=[1, 2, 3],
        events=[],
        required_actions=[],
        dependencies=[],
        transfer_rule={},
        initial_state_fingerprint="state",
        evaluator=lambda task, messages: {
            "db": len(messages) == 2,
            "communicate": [len(messages) >= 2],
        },
    )
    args.update(changes)
    return progress.build_progress_trace(**args)


def test_current_db_can_regress_while_communication_persists():
    result = build()
    assert result["phi"] == [0, 1, 0.5]
    assert result["check_values"] == [[False, False], [True, True], [False, True]]
    progress.validate_progress_trace(result, turns=2)


def test_initial_checks_omitted_from_both_numerator_and_denominator():
    result = build(
        evaluator=lambda task, messages: {
            "db": True,
            "communicate": [len(messages) >= 2],
        }
    )
    assert result["phi"] == [0, 1, 1]
    assert result["checks"][0]["included"] is False
    assert result["checks"][1]["included"] is True


def test_empty_checks_stay_zero_and_duplicate_prefix_is_not_re_evaluated():
    seen = []

    def evaluator(task, messages):
        seen.append(len(messages))
        return {"db": None, "communicate": []}

    result = build(task={"id": "0"}, evaluator=evaluator, prefix_lengths=[1, 1, 3])
    assert result["phi"] == [0, 0, 0]
    assert seen == [1, 3]


def test_required_actions_wait_for_predecessor_result_in_earlier_turn():
    actions = [
        {"action_id": "a", "name": "read_a", "arguments": {}},
        {"action_id": "b", "name": "read_b", "arguments": {}},
    ]
    events = [
        ToolEvent(event_id="a", sequence=0, turn_id=1, name="read_a", success=True),
        ToolEvent(event_id="b", sequence=1, turn_id=1, name="read_b", success=True),
        ToolEvent(event_id="b2", sequence=2, turn_id=2, name="read_b", success=True),
    ]
    result = build(
        task={"id": "0"},
        events=events,
        required_actions=actions,
        dependencies=[["a", "b"]],
        evaluator=lambda *args: {"db": None, "communicate": []},
    )
    assert result["phi"] == [0, 0.5, 1]


def test_only_frozen_prefix_reaches_evaluator_and_inputs_are_not_mutated():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "secret at end"},
    ]
    before = deepcopy(messages)
    seen = []

    def evaluator(task, prefix):
        seen.append(deepcopy(prefix))
        prefix.clear()
        task.clear()
        return {"db": None, "communicate": []}

    build(
        task={"id": "0"},
        messages=messages,
        prefix_lengths=[1, 1, 2],
        evaluator=evaluator,
    )
    assert seen[0] == before[:1]
    assert messages == before


@pytest.mark.parametrize("lengths", [[1, 3, 2], [1, 4], [True, 2], []])
def test_bad_prefix_boundary_is_not_partial_credit(lengths):
    with pytest.raises(ValueError):
        build(prefix_lengths=lengths)


def test_tampered_phi_and_checkset_are_rejected():
    result = build()
    result["phi"][-1] = 1
    with pytest.raises(ValueError, match="progress"):
        progress.validate_progress_trace(result, turns=2)
    result = build()
    result["checks"][0]["included"] = False
    with pytest.raises(ValueError):
        progress.validate_progress_trace(result, turns=2)


def test_ambiguous_transfer_target_rejected_before_scoring():
    with pytest.raises(ValueError, match="transfer"):
        build(transfer_rule={"allowed": True, "required": False})


def test_required_transfer_uses_fixed_checks_not_smaller_normal_denominator():
    event = ToolEvent(
        event_id="transfer",
        sequence=0,
        turn_id=1,
        name="transfer_to_human_agents",
        success=True,
    )
    result = build(
        events=[event],
        transfer_rule={
            "allowed": True,
            "required": True,
            "required_communication_checks": ["done"],
        },
    )
    assert result["phi"] == [0, 0.5, 1]
