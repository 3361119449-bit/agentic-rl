import asyncio
import json
from pathlib import Path

import httpx
import pytest

from test_trajectory_once_judge import frozen_inputs, mock_client
from tau2_agentic_rl.judge.routed import (
    MODE,
    CompactResult,
    expand,
    route,
    scored_trace,
)
from tau2_agentic_rl.judge.trajectory_once import evaluate_scoring_inputs


def inputs():
    x = frozen_inputs()
    x["reward_project_config"]["judge"]["mode"] = MODE
    x["judge"]["task"] = {"id": "15"}
    x["judge"]["transfer_validation_version"] = "transfer-policy-v2"
    x["routing_rule"] = {"task_id": "15", "mode": "always", "hints": []}
    return x


def answer():
    return dict(
        semantic_checks=[],
        transfer_semantic_checks=[],
        transfer_check=dict(
            applicable=False,
            valid=False,
            policy_required=False,
            self_caused_only=False,
            short_reason="No transfer was requested.",
        ),
        turns=[
            dict(message_index=2, verdict="neutral"),
            dict(
                message_index=4,
                verdict="bad",
                category="policy",
                target_quote="The booking price is $120.",
                reason="The quoted amount violates grounding.",
                evidence_message_indices=[3, 4],
            ),
        ],
    )


def test_online_path_mocked_offline(monkeypatch, scratch_dir):
    x = inputs()
    calls = []

    def respond(req):
        calls.append(json.loads(req.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(answer())}}]}
        )

    client = mock_client(monkeypatch, scratch_dir, respond)
    result, raw, ph, ck = asyncio.run(evaluate_scoring_inputs(client, x))
    assert result.mandatory_policy_checks[0].violation_assistant_turn_ids == [2]
    assert "mandatory_policy_checks" not in calls[0]["messages"][1]["content"]
    trace = scored_trace(
        x["simple_turn_inputs"],
        [],
        {"violating_turns": [1], "attribution_complete": True},
        (raw, ph, ck),
    )
    assert [t["bad_turn"] for t in trace["turns"]] == [False, True]
    assert len(trace["judge_calls"]) == 1
    asyncio.run(evaluate_scoring_inputs(client, x))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure", ["future", "missing", "duplicate", "wrong_quote", "neutral_reason"]
)
def test_invalid_rows(failure):
    x = inputs()
    a = answer()
    if failure == "future":
        a["turns"][1]["evidence_message_indices"] = [5]
    if failure == "missing":
        a["turns"].pop()
    if failure == "duplicate":
        a["turns"].append(a["turns"][0])
    if failure == "wrong_quote":
        a["turns"][1]["target_quote"] = "A different claim entirely."
    if failure == "neutral_reason":
        a["turns"][0]["reason"] = "This is a bad turn."
    with pytest.raises(ValueError):
        expand(CompactResult.model_validate(a), x["simple_turn_inputs"], x["judge"])


def test_no_api_for_explicit_skip():
    x = inputs()
    x["routing_rule"]["mode"] = "skip"
    result, raw, ph, ck = asyncio.run(evaluate_scoring_inputs(None, x))
    assert (raw, ph, ck) == (None, None, None)
    assert not result.semantic_checks
    credit = {"violating_turns": [], "attribution_complete": True}
    trace = scored_trace(x["simple_turn_inputs"], [], credit, (raw, ph, ck))
    assert trace["judge_calls"] == [] and trace["decisions"] == []
    assert all(not t["evidence"] and not t["bad_turn"] for t in trace["turns"])
    x["judge"]["semantic_checks"] = [{"criterion_id": "needs-evidence"}]
    assert route(x)["call"]


@pytest.mark.parametrize("routing_mode, expected_calls", [("always", 1), ("skip", 0)])
def test_live_then_frozen_retry_and_training_reader(
    monkeypatch, scratch_dir, routing_mode, expected_calls
):
    from test_procredit_rollout import rollout_fixture
    from tau2_agentic_rl.procredit_runtime import credit_from_record
    from tau2_agentic_rl.scoring_retry import retry_scoring

    requests = []

    def respond(request):
        body = json.loads(json.loads(request.content)["messages"][1]["content"])
        requests.append(body)
        clean = answer()
        clean["turns"] = [
            dict(message_index=t["message_index"], verdict="neutral")
            for t in body["targets"]
        ]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(clean)}}]}
        )

    client = mock_client(monkeypatch, scratch_dir / "cache", respond)
    loop, _, record = rollout_fixture(
        scratch_dir / "run",
        credit_version="procredit-turn-v4",
        final_turn_rule="simple-turn-v2",
        judge_mode=MODE,
        actual_judge=client,
        natural_stop_at=2,
        hard_limit=2,
        task_score_mode="official_semantic_progress_v1",
        routing_rule={"task_id": "0", "mode": routing_mode, "hints": []},
    )
    expected = credit_from_record(record)
    assert len(requests) == expected_calls
    assert (record.judge_result is None) == (routing_mode == "skip")
    record.custom_reward = None
    record.metadata["failure_phase"] = "turn_labels"
    assert asyncio.run(retry_scoring(record, client, loop.store))
    assert len(requests) == expected_calls
    assert credit_from_record(record) == expected
    record.metadata["judge_routing"]["call"] = not record.metadata["judge_routing"][
        "call"
    ]
    with pytest.raises(ValueError, match="routed scoring evidence"):
        credit_from_record(record)


def test_train30_table_and_profile():
    from tau2_agentic_rl.annotations import load_task_mapping
    from tau2_agentic_rl.config import load_yaml, validate_procredit_config
    from tau2_agentic_rl.judge.routed import validate_rule

    root = Path(__file__).resolve().parents[1]
    profile = load_yaml(
        root / "configs/rl/airline_procredit_v4_train30_c32_routed.yaml"
    )
    validate_procredit_config(profile)
    rules = load_task_mapping(root / profile["annotations"]["judge_routes"])
    tasks = json.loads(
        (root / profile["training_selection"]["split_file"]).read_bytes()
    )
    assert set(tasks["rl_train"]) == rules.keys()
    assert not set(tasks["official_test"]) & rules.keys()
    for rule in rules.values():
        validate_rule(rule)
    assert "task_judge_guidance" not in profile["annotations"]
    assert rules["40"]["mode"] == "name_correction"
    assert not rules["17"]["hints"] and rules["49"]["hints"]


@pytest.mark.parametrize("truncated", [False, True])
def test_rescore_skipped_record_without_transport(monkeypatch, scratch_dir, truncated):
    import sys
    from scripts import rescore_saved_trajectories as command
    from test_procredit_rollout import rollout_fixture
    from tau2_agentic_rl.procredit_runtime import credit_from_record
    from tau2_agentic_rl.storage import TrajectoryStore

    rule = {"task_id": "0", "mode": "skip", "hints": []}
    loop, _, record = rollout_fixture(
        scratch_dir / "live",
        credit_version="procredit-turn-v4",
        final_turn_rule="simple-turn-v2",
        task_score_mode="official_semantic_progress_v1",
        judge_mode=MODE,
        natural_stop_at=None if truncated else 2,
        hard_limit=2,
        zero_truncation=True,
        routing_rule=rule,
    )
    expected = credit_from_record(record)
    assert record.judge_result is None
    config = dict(
        loop.project,
        annotations={
            k: k
            for k in (
                "judge_routes",
                "required_actions",
                "action_dependencies",
                "transfer_rules",
                "semantic_checks",
                "policy_rules",
            )
        },
    )
    monkeypatch.setattr(command, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        command, "load_agent_system_prompt", lambda config, root: "fixture"
    )
    monkeypatch.setattr(
        command,
        "prompt_sha256",
        lambda prompt: record.metadata["agent_system_prompt_sha256"],
    )
    monkeypatch.setattr(
        command,
        "load_task_mapping",
        lambda path: {"0": rule if path.name == "judge_routes" else {}},
    )
    monkeypatch.setattr(command, "load_required_actions", lambda path: {"0": []})
    monkeypatch.setattr(command, "load_action_dependencies", lambda path: {})
    source, output = scratch_dir / "source", scratch_dir / "rescored"
    TrajectoryStore(source, attach_evaluation_identity=False).save(record)
    monkeypatch.setattr(
        sys, "argv", ["rescore", str(source), str(output), "--config", "fixture.yaml"]
    )
    command.main()
    saved = next(TrajectoryStore(output, attach_evaluation_identity=False).records())
    assert credit_from_record(saved) == expected


def test_refusal_is_distinct_from_policy_error():
    x = inputs()
    a = answer()
    row = a["turns"][1]
    row.update(
        verdict="refusal",
        reason="The prohibited operation was explicitly refused.",
        refusal_requests=[{"target": "booking", "operation": "cancel"}],
    )
    del row["category"]
    wire = json.dumps(a)
    result = expand(
        CompactResult.model_validate_json(wire), x["simple_turn_inputs"], x["judge"]
    )
    assert not result.mandatory_policy_checks
    credit = {"violating_turns": [], "attribution_complete": True}
    assert scored_trace(x["simple_turn_inputs"], [], credit, (wire, "prompt", "cache"))[
        "turns"
    ][1]["eligible_correct_refusal"]
    row["category"] = "policy"
    with pytest.raises(ValueError):
        expand(CompactResult.model_validate(a), x["simple_turn_inputs"], x["judge"])


def name_example():
    before = {
        "user_id": "anya_garcia_5901",
        "reservation_id": "3RK2T9",
        "passengers": [
            {"first_name": "Anya", "last_name": "Garcia", "dob": "1992-11-12"},
            {"first_name": "Mei", "last_name": "Lee", "dob": "1989-12-13"},
        ],
    }
    new = [before["passengers"][0], dict(before["passengers"][1], last_name="Garcia")]
    after = dict(before, passengers=new)
    events = []
    messages = []

    def add(role, content):
        messages.append({"role": role, "content": content})

    def tool(n, name, args, receipt):
        call = {"name": name, "arguments": args}
        body = json.dumps(receipt)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call_{n}", "function": call}],
                "raw_generated_text": "<tool_call>" + json.dumps(call) + "</tool_call>",
            }
        )
        add("tool", body)
        messages[-1]["tool_call_id"] = f"call_{n}"
        events.append(
            dict(turn_id=n, name=name, arguments=args, success=True, result=body)
        )

    add("user", "Hi, I need to change the passenger name on my reservation.")
    add(
        "assistant", "Could you please provide me with your user ID and reservation ID?"
    )
    add("user", "My user ID is anya_garcia_5901 and the reservation ID is 3RK2T9.")
    tool(2, "get_reservation_details", {"reservation_id": "3RK2T9"}, before)
    add(
        "assistant",
        "I found your reservation. It has two passengers: "
        "1. Anya Garcia (DOB 1992-11-12) 2. Mei Lee (DOB 1989-12-13). "
        "Which passenger would you like to change and what is the correct first name, last name, and date of birth?",
    )
    add(
        "user",
        "I want to change Mei Lee’s name to Mei Garcia. The date of birth stays the same.",
    )
    add(
        "assistant",
        "Here’s the change I’ll make: Passenger 2: Mei Lee → Mei Garcia (DOB 1989-12-13 unchanged). Shall I proceed?",
    )
    add("user", "Yes, please proceed.")
    tool(
        5,
        "update_reservation_passengers",
        {"reservation_id": "3RK2T9", "passengers": new},
        after,
    )
    add(
        "assistant",
        "Your passenger name has been successfully updated. Passenger 2 is now Mei Garcia (DOB 1989-12-13).",
    )
    return dict(
        routing_rule=dict(mode="name_correction", task_id="40", hints=[]),
        simple_turn_inputs={"messages": messages},
        judge=dict(
            task={"id": "40"},
            semantic_checks=[],
            transfer_rule={},
            trajectory=dict(tool_events=events, termination_reason="user_stop"),
        ),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "no_consent",
        "unrelated_text",
        "wrong_id",
        "wrong_dob",
        "extra_call",
        "mixed",
        "receipt",
        "transfer",
        "changed_prefix",
        "wrong_raw",
        "wrong_receipt_id",
    ],
)
def test_task40_fail_closed(fault):
    x = name_example()
    assert not route(x)["call"]
    m = x["simple_turn_inputs"]["messages"]
    events = x["judge"]["trajectory"]["tool_events"]
    if fault == "no_consent":
        m[8]["content"] = "Yes, please proceed?"
    elif fault == "unrelated_text":
        m[11]["content"] += " 另扣费"
    elif fault == "wrong_id":
        m[2]["content"] = m[2]["content"].replace("3RK2T9", "3rk2t9")
    elif fault == "wrong_dob":
        m[5]["content"] = m[5]["content"].replace("1992-11-12", "1989-12-13")
    elif fault == "extra_call":
        events.insert(0, dict(name="search_direct_flight"))
    elif fault == "mixed":
        m[9]["content"] = "A different passenger was added."
    elif fault == "receipt":
        m[10]["content"] = m[4]["content"]
    elif fault == "transfer":
        events.append(dict(name="transfer_to_human_agents"))
    elif fault == "changed_prefix":
        m[7]["content"] = "The change is already complete."
    elif fault == "wrong_raw":
        m[3]["raw_generated_text"] = "A different action" + m[3]["raw_generated_text"]
    elif fault == "wrong_receipt_id":
        m[4]["tool_call_id"] = "different"
    assert route(x)["call"]
