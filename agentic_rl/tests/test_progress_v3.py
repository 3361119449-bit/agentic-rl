"""Additive progress keeps independent verified goals visible in every branch."""

from copy import deepcopy

import pytest
from test_procredit_progress import build

from tau2_agentic_rl.reward.progress import validate_progress_trace
from tau2_agentic_rl.schemas import ToolEvent
from tau2_agentic_rl.versions import sha256_json


def action(action_id, reservation_id):
    return {
        "action_id": action_id,
        "name": "cancel_reservation",
        "arguments": {"reservation_id": reservation_id},
    }


def event(event_id, reservation_id, *, turn=1, sequence=0, **changes):
    args = dict(
        event_id=event_id, sequence=sequence, turn_id=turn,
        name="cancel_reservation", arguments={"reservation_id": reservation_id},
        success=True, db_effect=True,
    )
    args.update(changes)
    return ToolEvent(**args)


def additive(**changes):
    return build(version="progress-v3", **changes)


def test_db_communication_and_required_action_each_count():
    result = additive(
        required_actions=[action("write", "R")],
        events=[event("w", "R")],
    )
    assert [check["check_id"] for check in result["checks"]] == [
        "db", "comm:0", "req:write",
    ]
    assert all(check["included"] for check in result["checks"])
    assert result["phi"] == [0, 1, pytest.approx(2 / 3)]
    assert result["check_values"][-1] == [False, True, True]


def test_three_of_five_actions_progress_while_global_db_remains_false():
    result = additive(
        task={"id": "0"},
        required_actions=[action(str(i), str(i)) for i in range(5)],
        events=[event(str(i), str(i), sequence=i) for i in range(3)],
        evaluator=lambda *args: {"db": False, "communicate": []},
    )
    # One unfinished DB check plus five independent required actions.
    assert result["phi"] == [0, 0.5, 0.5]
    assert result["check_values"][-1] == [False, True, True, True, False, False]


def test_initial_db_and_communication_are_excluded_from_both_terms():
    result = additive(
        required_actions=[action("write", "R")],
        events=[event("w", "R")],
        evaluator=lambda *args: {"db": True, "communicate": [True]},
    )
    assert [check["included"] for check in result["checks"]] == [False, False, True]
    assert result["phi"] == [0, 1, 1]
    assert validate_progress_trace(result, turns=2) == [0, 1, 1]


def test_one_action_is_counted_once_despite_repeated_successful_calls():
    result = additive(
        task={"id": "0"},
        required_actions=[action("a", "A"), action("b", "B")],
        events=[event("a1", "A"), event("a2", "A", turn=2, sequence=1)],
        evaluator=lambda *args: {"db": None, "communicate": []},
    )
    assert result["phi"] == [0, 0.5, 0.5]


def test_one_event_cannot_complete_two_distinct_annotation_actions():
    result = additive(
        task={"id": "0"},
        required_actions=[action("a", "A"), action("b", "A")],
        events=[event("a1", "A"), event("a2", "A", turn=2, sequence=1)],
        evaluator=lambda *args: {"db": None, "communicate": []},
    )
    assert result["phi"] == [0, 0.5, 1]


@pytest.mark.parametrize("changes", [
    {"success": False},
    {"arguments": {"reservation_id": "WRONG"}},
    {"db_effect": False},
    {"db_effect": None},
])
def test_failed_wrong_argument_or_ineffective_write_never_completes_req(changes):
    result = additive(
        task={"id": "0"}, required_actions=[action("a", "A")],
        events=[event("a", "A", **changes)],
        evaluator=lambda *args: {"db": False, "communicate": []},
    )
    assert result["phi"] == [0, 0, 0]
    assert result["check_values"][-1] == [False, False]


def test_required_transfer_preserves_special_checks_and_adds_annotated_req():
    rule = {
        "allowed": True, "required": True,
        "required_pre_transfer_action_groups": [{
            "any_of": [{"tool": "cancel_reservation",
                        "arguments": {"reservation_id": "R"}}],
            "require_db_effect": True,
        }],
        "required_communication_checks": ["done"],
    }
    events = [
        event("w", "R"),
        ToolEvent(event_id="transfer", sequence=1, turn_id=2,
                  name="transfer_to_human_agents", success=True),
    ]
    result = additive(
        transfer_rule=rule, required_actions=[action("w", "R")], events=events,
    )
    assert [check["check_id"] for check in result["checks"]] == [
        "transfer_call", "transfer_pre:0", "transfer_comm:0", "req:w",
    ]
    assert result["phi"] == [0, 0.5, 1]
    assert result["check_values"][1] == [False, True, False, True]
    legacy = build(
        version="progress-v2", transfer_rule=rule,
        required_actions=[action("w", "R")], events=events,
    )
    assert [row[:3] for row in result["check_values"]] == legacy["check_values"]
    assert legacy["phi"] == [0, pytest.approx(1 / 3), 1]


@pytest.mark.parametrize("rule, expected_ids", [
    ({"allowed": True, "required": False}, ["db", "req:w"]),
    ({"allowed": False, "required": True}, ["transfer_call", "req:w"]),
])
def test_allowed_advisory_does_not_block_v3_progress(rule, expected_ids):
    result = additive(
        task={"id": "0"}, transfer_rule=rule,
        required_actions=[action("w", "R")], events=[event("w", "R")],
        evaluator=lambda *args: {"db": False, "communicate": []},
    )
    assert [check["check_id"] for check in result["checks"]] == expected_ids
    assert result["phi"] == [0, 0.5, 0.5]


def test_v3_rejects_suppressed_req_even_with_rehashed_consistent_trace():
    result = additive(
        required_actions=[action("w", "R")], events=[event("w", "R")],
    )
    result["checks"][-1]["included"] = False
    result["definition"]["checks"] = deepcopy(result["checks"])
    result["checkset_fingerprint"] = sha256_json(result["definition"])
    result["phi"] = [0, 1, 0.5]
    with pytest.raises(ValueError, match="applicability"):
        validate_progress_trace(result, turns=2)


@pytest.mark.parametrize("tamper", ["unknown", "definition", "matcher", "downgrade"])
def test_v3_rejects_tampered_version_identity(tamper):
    result = additive(
        required_actions=[action("w", "R")], events=[event("w", "R")],
    )
    if tamper == "unknown":
        result["version"] = "progress-v4"
    elif tamper == "definition":
        result["definition"]["version"] = "progress-v2"
    elif tamper == "matcher":
        result["definition"]["matcher_version"] = "wrong-matcher"
    else:
        result["version"] = result["definition"]["version"] = "progress-v2"
    result["checkset_fingerprint"] = sha256_json(result["definition"])
    with pytest.raises(ValueError):
        validate_progress_trace(result, turns=2)
