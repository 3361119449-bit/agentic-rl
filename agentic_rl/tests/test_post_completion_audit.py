import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audit_db_subgoals import load_fixed_tools, replay_initial
from scripts.audit_post_completion import fixed_communicate_verifier, verify_db_prefixes
from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.schemas import ToolEvent


@pytest.fixture
def native_record():
    root = os.environ.get("DB_SUBGOAL_TAU2_SOURCE")
    if not root:
        pytest.skip("set DB_SUBGOAL_TAU2_SOURCE for fixed Tau2 replay")
    source = Path(root)
    cls, db = load_fixed_tools(source)
    tool = cls(db.model_copy(deep=True))
    bag = dict(
        reservation_id="YAX4DR",
        total_baggages=2,
        nonfree_baggages=0,
        payment_id="credit_card_4938634",
    )
    good = tool.update_reservation_baggages(**bag).model_dump_json()
    flights = [
        {k: v for k, v in f.model_dump().items() if k in {"flight_number", "date"}}
        for f in tool.db.reservations["YAX4DR"].flights
    ]
    wrong = dict(
        reservation_id="YAX4DR",
        cabin="business",
        flights=flights,
        payment_id="credit_card_4938634",
    )
    bad = tool.update_reservation_flights(**wrong).model_dump_json()
    events = [
        ToolEvent(
            event_id="good",
            sequence=0,
            turn_id=1,
            name="update_reservation_baggages",
            arguments=bag,
            success=True,
            db_effect=True,
            result=good,
        ),
        ToolEvent(
            event_id="bad",
            sequence=1,
            turn_id=3,
            name="update_reservation_flights",
            arguments=wrong,
            success=True,
            db_effect=True,
            result=bad,
        ),
    ]
    task = dict(
        id="12",
        evaluation_criteria=dict(
            reward_basis=["DB", "COMMUNICATE"],
            communicate_info=[],
            actions=[dict(name="update_reservation_baggages", arguments=bag)],
        ),
    )
    messages = [dict(role="assistant", content=str(i)) for i in range(3)]
    trace = build_progress_trace(
        task=task,
        messages=messages,
        prefix_lengths=[0, 1, 2, 3],
        events=events,
        required_actions=[],
        dependencies=[],
        transfer_rule={},
        initial_state_fingerprint="i",
        evaluator=lambda task, prefix: dict(
            db=[False, True, True, False][len(prefix)], communicate=[]
        ),
        version="progress-v2",
    )
    record = SimpleNamespace(
        trajectory_id="native12",
        scoring_inputs=dict(judge=dict(task=task)),
        progress_trace=trace,
        tool_events=events,
        assistant_turns=3,
        environment_transcript=messages,
    )
    return source, cls, db, record


def test_fixed_full_db_replay_proves_completion_then_destruction(native_record):
    source, cls, db, record = native_record
    assert verify_db_prefixes(record, cls, db) == 4
    assert fixed_communicate_verifier(source)(record) == 3


def test_fixed_replay_rejects_an_invented_success_prefix(native_record):
    _, cls, db, record = native_record
    record.progress_trace["check_values"][2][0] = False
    with pytest.raises(ValueError, match="DB prefixes"):
        verify_db_prefixes(record, cls, db)


def test_locally_rejected_multi_tool_turn_is_never_reexecuted(native_record):
    _, cls, db, record = native_record
    record.tool_events = [
        record.tool_events[0].model_copy(
            update={
                "success": False,
                "db_effect": None,
                "error_kind": "multiple_tool_calls",
                "result": "rejected mixed done turn",
            }
        )
    ]
    for state in record.progress_trace["check_values"]:
        state[0] = False
    assert verify_db_prefixes(record, cls, db) == 4


def test_shared_db_replay_also_skips_locally_rejected_multi_tool_turn(native_record):
    _, cls, db, record = native_record
    from tau2_agentic_rl.reward.required_actions import load_required_actions

    required = load_required_actions(
        Path(__file__).parents[1]
        / "data/annotations/airline_required_actions.train.v1.json"
    )["33"]
    record.scoring_inputs = {
        "judge": {
            "task": {"id": "33", "evaluation_criteria": {"reward_basis": ["DB"]}},
            "transfer_rule": {},
        },
        "required_actions": required,
        "action_dependencies": [],
    }
    record.tool_events = [
        ToolEvent(
            event_id="blocked",
            sequence=0,
            turn_id=1,
            name=required[0]["name"],
            arguments=required[0]["arguments"],
            error_kind="multiple_tool_calls",
            success=False,
            db_effect=None,
            result="rejected",
        )
    ]
    initial, checked, error = replay_initial(record, cls, db)
    assert initial and checked == 0 and error is None


def test_fixed_communicate_checks_assistant_text_and_every_prefix(native_record):
    source, _, _, record = native_record
    record = deepcopy(record)
    record.scoring_inputs["judge"]["task"]["evaluation_criteria"][
        "communicate_info"
    ] = ["done"]
    record.environment_transcript = [
        dict(role="user", content="done"),
        dict(role="assistant", content="Done"),
        dict(role="assistant", content="query"),
    ]
    record.progress_trace["checks"].append(dict(check_id="comm:0"))
    for state, value in zip(
        record.progress_trace["check_values"], [False, False, True, True], strict=True
    ):
        state.append(value)
    assert fixed_communicate_verifier(source)(record) == 3
    record.progress_trace["check_values"][1][-1] = True
    with pytest.raises(ValueError, match="COMMUNICATE"):
        fixed_communicate_verifier(source)(record)
