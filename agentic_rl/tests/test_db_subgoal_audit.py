import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_procredit_rollout import rollout_fixture

from scripts.audit_db_subgoals import compare_group, load_fixed_tools, replay_initial
from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.procredit_runtime import credit_from_record, queue_group_reports
from tau2_agentic_rl.reward.required_actions import load_required_actions
from tau2_agentic_rl.schemas import ToolEvent


def report_fixture(scratch_dir):
    _, _, record = rollout_fixture(scratch_dir, credit_version="procredit-turn-v4")
    records = [
        record.model_copy(update={"trajectory_id": f"{i:032x}"}) for i in range(8)
    ]
    report = queue_group_reports(
        [f"group_{i}_0" for i in range(8)],
        [{"procredit": credit_from_record(r)} for r in records],
        CreditConfig(version="procredit-turn-v4"),
    )["group"]
    return report, records


def test_audit_rejects_changed_original_membership(scratch_dir):
    report, records = report_fixture(scratch_dir)
    report["members"][0] = "other_0_0"
    with pytest.raises(ValueError, match="original|membership"):
        compare_group(report, records, lambda r: ({}, 0, None))


def test_audit_preserves_source_records_and_compares_every_policy_token(scratch_dir):
    report, records = report_fixture(scratch_dir)
    source_report = deepcopy(report)
    source_records = [r.model_dump() for r in records]
    result = compare_group(report, records, lambda r: ({}, 0, None))
    assert result["misrewarded_turns"] == []
    assert all(t["delta"] == 0 for r in result["trajectories"] for t in r["turns"])
    assert source_records == [r.model_dump() for r in records]
    assert report == source_report


def test_audit_rejects_a_mismatched_baseline_advantage(scratch_dir):
    report, records = report_fixture(scratch_dir)
    report["credit"]["trajectories"][0]["token_advantages"][0] += 0.01
    with pytest.raises(ValueError, match="original"):
        compare_group(report, records, lambda r: ({}, 0, None))


@pytest.fixture
def fixed_airline():
    source = os.environ.get("DB_SUBGOAL_TAU2_SOURCE")
    if not source:
        pytest.skip("set DB_SUBGOAL_TAU2_SOURCE for fixed Tau2 method replay")
    cls, db = load_fixed_tools(Path(source))
    required = load_required_actions(
        Path(__file__).parents[1]
        / "data/annotations/airline_required_actions.train.v1.json"
    )["33"]
    inputs = {
        "judge": {
            "task": {"id": "33", "evaluation_criteria": {"reward_basis": ["DB"]}},
            "transfer_rule": {},
        },
        "required_actions": required,
        "action_dependencies": [],
    }
    return cls, db, inputs


def test_fixed_replay_handles_extra_cancellation_and_matches_live_snapshot(
    fixed_airline,
):
    cls, db, inputs = fixed_airline
    tools = cls(db.model_copy(deep=True))
    result = tools.cancel_reservation("HXDUBJ")
    e = ToolEvent(
        event_id="cancel",
        sequence=0,
        turn_id=1,
        name="cancel_reservation",
        arguments={"reservation_id": "HXDUBJ"},
        success=True,
        db_effect=True,
        result=result.model_dump_json(),
    )
    initial, checked, error = replay_initial(
        SimpleNamespace(scoring_inputs=inputs, tool_events=[e]), cls, db
    )
    assert checked == 1 and error is None
    assert initial["HXDUBJ"]["status"] is None


def test_fixed_replay_rejects_stale_pre_call_payment_context(fixed_airline):
    cls, db, inputs = fixed_airline
    a = inputs["required_actions"][1]
    tools = cls(db.model_copy(deep=True))
    result = tools.update_reservation_baggages(**a["arguments"])
    e = ToolEvent(
        event_id="bags",
        sequence=0,
        turn_id=1,
        name=a["name"],
        arguments=a["arguments"],
        success=True,
        db_effect=True,
        result=json.dumps(result.model_dump(mode="json")),
        state_before={"reservation_id": "HXDUBJ", "nonfree_baggages": 999},
    )
    initial, checked, error = replay_initial(
        SimpleNamespace(scoring_inputs=inputs, tool_events=[e]), cls, db
    )
    assert not initial and checked == 0
    assert "state_before" in error
