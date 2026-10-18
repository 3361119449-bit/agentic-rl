import json
from copy import deepcopy

import pytest

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.schemas import ToolEvent


def case(db=None, comm=None, *, basis=None):
    db = db or [False, True, True, False]
    turns = len(db) - 1
    task = {
        "id": "t",
        "evaluation_criteria": {
            "reward_basis": basis or ["DB", "COMMUNICATE"],
            "communicate_info": ["done"] if comm is not None else [],
        },
    }
    actions = [
        {
            "action_id": "bags",
            "name": "update_reservation_baggages",
            "arguments": {
                "reservation_id": "R",
                "total_baggages": 2,
                "nonfree_baggages": 0,
                "payment_id": "card",
            },
        }
    ]
    events = [
        ToolEvent(
            event_id="good",
            sequence=0,
            turn_id=1,
            name=actions[0]["name"],
            arguments=actions[0]["arguments"],
            success=True,
            db_effect=True,
        ),
        ToolEvent(
            event_id="bad",
            sequence=1,
            turn_id=turns,
            name="update_reservation_flights",
            arguments={"reservation_id": "R", "cabin": "business"},
            success=True,
            db_effect=True,
        ),
    ]
    messages = [{"role": "assistant", "content": str(i)} for i in range(turns)]
    progress = build_progress_trace(
        task=task,
        messages=messages,
        prefix_lengths=list(range(turns + 1)),
        events=events,
        required_actions=actions,
        dependencies=[],
        transfer_rule={},
        initial_state_fingerprint="initial",
        evaluator=lambda task, prefix: {
            "db": db[len(prefix)],
            "communicate": [comm[len(prefix)]] if comm is not None else [],
        },
        version="progress-v2",
    )
    return dict(
        task=task,
        progress_trace=progress,
        events=events,
        turns=turns,
        required_actions=actions,
        transfer_rule={},
        violating_turns=[],
        termination_reason="user_stop",
        subgoal_trace=None,
    )


def build(data=None, **changes):
    from tau2_agentic_rl.reward.post_completion import build_post_completion_trace

    return build_post_completion_trace(**((data or case()) | changes))


def row(data, i=0, *, bad=None):
    terminal = int(data["progress_trace"]["check_values"][-1][0])
    phi = data["progress_trace"]["phi"]
    bad = bad or []
    return dict(
        trajectory_id=f"r{i}",
        task_id="t",
        policy_version=0,
        checkset_fingerprint=data["progress_trace"]["checkset_fingerprint"],
        initial_state_fingerprint="initial",
        score=terminal + 0.5 * phi[-1],
        terminal_success=terminal,
        multiplier=1,
        phi=phi,
        response_turn_ids=[-1, *range(data["turns"])],
        valid=not bad,
        credit_version="procredit-turn-v4",
        policy_credit=dict(
            version="policy-credit-v1",
            violating_turns=bad,
            attribution_complete=True,
            unresolved_checks=[],
        ),
        process_credit=dict(
            version="process-credit-v2", turn_costs=[0.08] * data["turns"]
        ),
    )


def test_completion_bonus_and_destruction_ceiling_are_local():
    data = case()
    r = row(data)
    old = compute_group_credit([r], CreditConfig(version="procredit-turn-v4"))
    r["completion_trace"] = build(data)
    cfg = CreditConfig(
        version="procredit-turn-v4",
        post_completion_credit="post_completion_regression_v1",
    )
    new = compute_group_credit([r], cfg)
    a, b = old["trajectories"][0], new["trajectories"][0]
    assert r["completion_trace"]["first_success_turn"] == 1
    assert r["completion_trace"]["regression_turns"] == [3]
    assert b["token_advantages"][1] - a["token_advantages"][1] == pytest.approx(0.5)
    assert b["token_advantages"][2] == a["token_advantages"][2]
    assert b["token_advantages"][3] <= -1.08
    assert b["token_advantages"][0] == 0
    assert new["turn_mean"] == old["turn_mean"]
    assert new["score_mean"] == old["score_mean"]
    assert b["trajectory_advantage"] == a["trajectory_advantage"]


def test_all_official_communication_checks_are_required_even_if_phi_looks_complete():
    data = case(comm=[False, False, False, True])
    trace = build(data)
    assert trace["first_success_turn"] is None
    assert trace["regression_turns"] == []
    assert trace["turn_bonus"] == [0, 0, 0]


@pytest.mark.parametrize(
    "basis", [["DB", "NL_ASSERTION"], ["DB", "ACTION"], ["DB", "ENV_ASSERTION"]]
)
def test_unverifiable_official_conditions_disable_completion_credit(basis):
    trace = build(case(basis=basis))
    assert not trace["applicable"]
    assert trace["turn_bonus"] == [0, 0, 0]


@pytest.mark.parametrize("db", [[False, True, True, True], [False, False, False, True]])
def test_final_success_and_normal_queries_do_not_get_extra_reward_without_regression(
    db,
):
    trace = build(case(db))
    assert trace["regression_turns"] == []
    assert trace["turn_bonus"] == [0, 0, 0]


def test_recovery_does_not_reward_success_repeatedly():
    data = case([False, True, False, True, False, True])
    data["events"] = [data["events"][0]] + [
        ToolEvent(
            event_id=f"e{i}",
            sequence=i,
            turn_id=t,
            name="update_reservation_flights",
            arguments={"reservation_id": "R"},
            success=True,
            db_effect=True,
        )
        for i, t in enumerate([2, 4], 1)
    ]
    trace = build(data)
    assert trace["first_success_turn"] == 1
    assert trace["regression_turns"] == [2, 4]
    assert trace["turn_bonus"] == [0.5, 0, 0, 0, 0]


def test_initial_satisfaction_gets_no_bonus_but_real_destruction_is_negative():
    trace = build(case([True, True, True, False]))
    assert trace["initial_success"]
    assert trace["first_success_turn"] is None
    assert trace["positive_floor_turn"] is None
    assert trace["turn_bonus"] == [0, 0, 0]
    assert trace["regression_turns"] == [3]


@pytest.mark.parametrize("kind", ["read", "unknown_effect", "external", "user_tools"])
def test_no_penalty_for_unattributed_external_or_unverified_db_drop(kind):
    data = case()
    if kind == "read":
        data["events"][-1].name = "get_reservation_details"
        data["events"][-1].db_effect = False
    elif kind == "unknown_effect":
        data["events"][-1].db_effect = None
    elif kind == "external":
        data["termination_reason"] = "infrastructure_error"
    else:
        data["task"]["user_tools"] = ["external_mutation"]
    trace = build(data)
    assert trace["regression_turns"] == []
    assert not any(trace["turn_bonus"])


def test_same_turn_policy_penalty_is_not_added_twice():
    data = case()
    r = row(data, bad=[2])
    old = compute_group_credit([r], CreditConfig(version="procredit-turn-v4"))
    r["completion_trace"] = build(data, violating_turns=[2])
    cfg = CreditConfig(
        version="procredit-turn-v4",
        post_completion_credit="post_completion_regression_v1",
    )
    new = compute_group_credit([r], cfg)
    assert (
        new["trajectories"][0]["token_advantages"][-1]
        == old["trajectories"][0]["token_advantages"][-1]
    )


def test_high_trajectory_advantage_cannot_cancel_destruction_limit_after_recovery():
    data = case([False, True, False, True])
    data["events"][-1].turn_id = 2
    group = [row(data)] + [
        row(case([False, False, False, False]), i) for i in range(1, 8)
    ]
    for r in group:
        r["completion_trace"] = build(
            data if r["trajectory_id"] == "r0" else case([False, False, False, False])
        )
    out = compute_group_credit(
        group,
        CreditConfig(
            version="procredit-turn-v4",
            post_completion_credit="post_completion_regression_v1",
        ),
    )
    assert out["trajectories"][0]["trajectory_advantage"] > 1
    assert out["trajectories"][0]["token_advantages"][2] <= -1.08


def test_mixed_first_completion_turn_with_wrong_write_cannot_receive_bonus():
    data = case()
    data["events"].insert(
        1,
        ToolEvent(
            event_id="extra",
            sequence=1,
            turn_id=1,
            name="send_certificate",
            arguments={"user_id": "u"},
            success=True,
            db_effect=True,
        ),
    )
    data["events"][-1].sequence = 2
    trace = build(data)
    assert trace["turn_bonus"] == [0, 0, 0]
    assert trace["positive_floor_turn"] is None


def test_db_positive_credit_and_completion_credit_share_the_half_point_cap():
    from test_db_subgoals import build as db_build
    from test_db_subgoals import event, inputs

    d = inputs()
    a, b = d["required_actions"]
    first = event(a)
    second = event(b, turn=2, sequence=1, state=json.loads(first.result))
    subgoals = db_build([first, second], turns=3)
    trace = build(case([False, False, True, False]), subgoal_trace=subgoals)
    assert subgoals["turn_credits"] == [0.25, 0.25, 0]
    assert trace["turn_bonus"] == [0, 0, 0]


def test_policy_failure_on_first_success_blocks_bonus_but_not_destruction_limit():
    trace = build(violating_turns=[0])
    assert not any(trace["turn_bonus"])
    assert trace["positive_floor_turn"] is None
    assert trace["regression_turns"] == [3]
    r = row(case(), bad=[0]) | {"completion_trace": trace}
    out = compute_group_credit(
        [r],
        CreditConfig(
            version="procredit-turn-v4",
            post_completion_credit="post_completion_regression_v1",
        ),
    )["trajectories"][0]
    assert out["token_advantages"][1] <= -1.08


def test_unobserved_success_and_destruction_inside_one_tool_batch_cannot_earn_credit():
    data = case([False, False, False, False])
    data["events"][-1].turn_id = 1
    trace = build(data)
    assert trace["first_success_turn"] is None
    assert trace["regression_turns"] == []


def test_partial_db_positive_credit_only_leaves_the_remaining_completion_budget():
    from test_db_subgoals import build as db_build
    from test_db_subgoals import event, inputs

    d = inputs()
    d["initial_reservations"]["R"]["cabin"] = "economy"
    e = event(d["required_actions"][1], state=d["initial_reservations"]["R"])
    subgoals = db_build([e], **(d | {"turns": 3}))
    trace = build(subgoal_trace=subgoals)
    assert subgoals["turn_credits"] == [0.25, 0, 0]
    assert trace["turn_bonus"] == [0.25, 0, 0]


def test_actual_tensor_padding_and_observation_advantages_remain_zero():
    torch = pytest.importorskip("torch")
    from tau2_agentic_rl.procredit_runtime import build_credit_tensors

    data = case()
    extras = [
        {"procredit": row(data, i) | {"completion_trace": build(data)}}
        for i in range(8)
    ]
    keys = [f"g_{i}_0" for i in range(8)] + ["pad_0_0"]
    extras.append(deepcopy(extras[0]))
    fields, _ = build_credit_tensors(
        keys,
        extras,
        torch.tensor([[0, 1, 1, 1]] * 8 + [[0] * 4]),
        CreditConfig(
            version="procredit-turn-v4",
            post_completion_credit="post_completion_regression_v1",
        ),
        padding=[False] * 8 + [True],
    )
    advantages = fields["advantages"]
    assert advantages[0, -1] <= -1.08
    assert torch.count_nonzero(advantages[-1]) == 0
    assert torch.count_nonzero(advantages[:, 0]) == 0


def test_disabled_configuration_keeps_the_prior_audit_shape():
    config = CreditConfig(version="procredit-turn-v4")
    assert "post_completion_credit" not in config.audit_config()
    with pytest.raises(ValueError):
        CreditConfig(post_completion_credit="post_completion_regression_v1")
    with pytest.raises(ValueError):
        CreditConfig(version="procredit-turn-v4", post_completion_credit="unknown")


@pytest.mark.parametrize("process_cost", [0.0, 0.08, 0.5, 1.2])
def test_first_completion_final_advantage_is_positive_after_all_local_costs(
    process_cost,
):
    data = case()
    bad = row(data)
    good_data = case([False, False, False, True])
    good = row(good_data, 1)
    bad["process_credit"]["turn_costs"][0] = process_cost
    bad["completion_trace"] = build(data)
    good["completion_trace"] = build(good_data)
    cfg = CreditConfig(
        version="procredit-turn-v4",
        post_completion_credit="post_completion_regression_v1",
    )
    result = compute_group_credit([bad, good], cfg)["trajectories"][0]
    assert result["trajectory_advantage"] < 0
    assert result["token_advantages"][1] == pytest.approx(max(1e-6, 0.5 - process_cost))
    assert result["token_advantages"][1] > 0
    assert result["completion_turn_floor"][0] == result["token_advantages"][1]
    assert result["completion_turn_floor_adjustments"][0] > 0
    assert result["token_advantages"][-1] <= -1.08


def test_exhausted_db_reward_budget_cannot_disable_the_first_completion_sign():
    from test_db_subgoals import build as db_build
    from test_db_subgoals import event, inputs

    d = inputs()
    a, b = d["required_actions"]
    first = event(a)
    second = event(b, turn=2, sequence=1, state=json.loads(first.result))
    subgoals = db_build([first, second], turns=3)
    data = case([False, False, True, False])
    r = row(data)
    r["subgoal_trace"] = subgoals
    r["completion_trace"] = build(data, subgoal_trace=subgoals)
    cfg = CreditConfig(
        version="procredit-turn-v4",
        db_subgoal_credit="db_subgoal_local_v1",
        post_completion_credit="post_completion_regression_v1",
    )
    item = compute_group_credit([r], cfg)["trajectories"][0]
    assert r["completion_trace"]["completion_positive_credit"] == 0
    assert r["completion_trace"]["db_positive_credit"] == 0.5
    assert item["token_advantages"][2] >= 0.42 - 1e-12
    assert item["token_advantages"][2] > 0


def test_first_completion_floor_is_a_maximum_not_a_second_additive_award():
    data = case([False, True, False, True])
    data["events"][-1].turn_id = 2
    group = [row(data)] + [
        row(case([False, False, False, False]), i) for i in range(1, 8)
    ]
    cfg = CreditConfig(
        version="procredit-turn-v4",
        post_completion_credit="post_completion_regression_v1",
    )
    for i, r in enumerate(group):
        r["completion_trace"] = build(
            data if i == 0 else case([False, False, False, False])
        )
    baseline = compute_group_credit(group, CreditConfig(version="procredit-turn-v4"))
    current = compute_group_credit(group, cfg)
    before = baseline["trajectories"][0]["token_advantages"][1]
    after = current["trajectories"][0]["token_advantages"][1]
    assert before > 0.5
    assert after == pytest.approx(before + 0.5)
    assert current["trajectories"][0]["completion_turn_floor_adjustments"][0] == 0


def test_completion_sign_cannot_silently_be_lost_to_missing_turn_tokens():
    r = row(case()) | {"completion_trace": build()}
    r["response_turn_ids"] = [-1, -1, 1, 2]
    with pytest.raises(
        ValueError, match="first completion turn has no trainable tokens"
    ):
        compute_group_credit(
            [r],
            CreditConfig(
                version="procredit-turn-v4",
                post_completion_credit="post_completion_regression_v1",
            ),
        )


@pytest.mark.parametrize(
    "change", [{"positive_floor_turn": 2}, {"positive_margin": 0.0}]
)
def test_refingerprinted_trace_cannot_change_completion_sign_contract(change):
    from tau2_agentic_rl.reward.post_completion import validate_post_completion_trace
    from tau2_agentic_rl.versions import sha256_json

    trace = build() | change
    trace["trace_fingerprint"] = sha256_json(
        {k: v for k, v in trace.items() if k != "trace_fingerprint"}
    )
    with pytest.raises(ValueError, match="completion"):
        validate_post_completion_trace(trace, turns=3, violating_turns=[])


def test_refingerprinted_trace_cannot_claim_official_success_with_a_failed_db():
    from tau2_agentic_rl.reward.post_completion import validate_post_completion_trace
    from tau2_agentic_rl.versions import sha256_json

    trace = build()
    trace["db_success"][1] = False
    trace["trace_fingerprint"] = sha256_json(
        {k: v for k, v in trace.items() if k != "trace_fingerprint"}
    )
    with pytest.raises(ValueError, match="completion"):
        validate_post_completion_trace(trace, turns=3, violating_turns=[])


def test_live_scoring_and_frozen_retry_share_completion_trace(scratch_dir):
    import asyncio

    from test_procredit_rollout import rollout_fixture

    from tau2_agentic_rl.procredit_runtime import credit_from_record
    from tau2_agentic_rl.scoring_retry import retry_scoring

    loop, output, record = rollout_fixture(
        scratch_dir,
        credit_version="procredit-turn-v4",
        post_completion_credit=True,
    )
    expected = deepcopy(record.completion_trace)
    assert output.extra_fields["procredit"]["completion_trace"] == expected
    assert not expected["applicable"]  # This fixture is COMM-only.
    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert record.completion_trace == expected
    record.completion_trace["turn_bonus"][0] = 0.5
    with pytest.raises(ValueError, match="completion"):
        credit_from_record(record)
