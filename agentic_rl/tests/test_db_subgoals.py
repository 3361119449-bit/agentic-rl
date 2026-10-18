import json
from copy import deepcopy

import pytest

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.schemas import ToolEvent


def inputs():
    reservation = dict(
        reservation_id="R",
        cabin="basic_economy",
        flights=[dict(flight_number="F", date="2024-05-25")],
        passengers=[dict(first_name="Old", last_name="Name", dob="1970-01-01")],
        total_baggages=0,
        nonfree_baggages=0,
        status=None,
    )
    actions = [
        dict(
            action_id="flight",
            name="update_reservation_flights",
            arguments=dict(
                reservation_id="R",
                cabin="economy",
                flights=deepcopy(reservation["flights"]),
                payment_id="gift",
            ),
        ),
        dict(
            action_id="bags",
            name="update_reservation_baggages",
            arguments=dict(
                reservation_id="R",
                total_baggages=3,
                nonfree_baggages=1,
                payment_id="gift",
            ),
        ),
    ]
    return dict(
        task={"id": "t", "evaluation_criteria": {"reward_basis": ["DB"]}},
        required_actions=actions,
        dependencies=[],
        transfer_rule={},
        initial_reservations={"R": reservation},
        initial_state_fingerprint="initial",
        violating_turns=[],
        turns=5,
    )


def event(action, *, turn=1, sequence=0, state=None, **changes):
    snapshot = deepcopy(
        inputs()["initial_reservations"]["R"] if state is None else state
    )
    snapshot.update({k: v for k, v in action["arguments"].items() if k != "payment_id"})
    return ToolEvent(
        event_id=f"e{sequence}",
        sequence=sequence,
        turn_id=turn,
        name=action["name"],
        arguments=deepcopy(action["arguments"]),
        success=True,
        db_effect=True,
        result=json.dumps(snapshot),
        **changes,
    )


def build(events, **changes):
    from tau2_agentic_rl.reward.db_subgoals import build_db_subgoal_trace

    return build_db_subgoal_trace(events=events, **(inputs() | changes))


def test_partial_success_is_local_and_later_wrong_write_cannot_receive_it():
    a, b = inputs()["required_actions"]
    good = event(b, turn=2)
    wrong = deepcopy(a)
    wrong["arguments"]["cabin"] = "business"
    trace = build(
        [good, event(wrong, turn=4, sequence=1, state=json.loads(good.result))]
    )
    assert trace["phi"] == [0, 0, 0.5, 0.5, 0.5, 0.5]
    assert trace["turn_credits"] == [0, 0.25, 0, 0, 0]
    assert trace["wrong_write_turns"] == [3]


def test_mixed_turn_gets_no_positive_credit_or_deferred_reward_on_query():
    a, b = inputs()["required_actions"]
    b["arguments"]["nonfree_baggages"] = 2
    good = event(a)
    trace = build([good, event(b, sequence=1, state=json.loads(good.result))])
    assert trace["phi"][1] == 0.5
    assert trace["turn_credits"] == [0] * 5
    assert trace["deferred_turns"] == [0]


@pytest.mark.parametrize(
    "mutate", ["payment", "order", "snapshot", "failure", "unknown"]
)
def test_success_flag_or_similar_fields_do_not_replace_full_evidence(mutate):
    a = inputs()["required_actions"][0]
    e = event(a)
    if mutate == "payment":
        e.arguments["payment_id"] = "wrong"
    elif mutate == "order":
        e.arguments["reservation_id"] = "other"
    elif mutate == "snapshot":
        e.result = "{}"
    elif mutate == "failure":
        e.success = False
    else:
        e.db_effect = None
    assert build([e])["turn_credits"] == [0] * 5


def test_regression_is_local_bounded_and_repair_cannot_farm_credit():
    a = inputs()["required_actions"][0]
    wrong = deepcopy(a)
    wrong["arguments"]["cabin"] = "business"
    trace = build(
        [
            event(x, turn=i + 1, sequence=i)
            for i, x in enumerate([a, wrong, a, wrong, a])
        ]
    )
    assert trace["turn_credits"] == [0.25, -0.25, 0, 0, 0]
    assert trace["regression_turns"] == [1, 3]


def test_initial_satisfaction_cannot_be_rewarded_and_policy_blocks_only_its_turn():
    data = inputs()
    a = data["required_actions"][0]
    data["initial_reservations"]["R"]["cabin"] = "economy"
    assert build([event(a)], **data)["turn_credits"] == [0] * 5
    assert build([event(a)], violating_turns=[0])["turn_credits"] == [0] * 5


@pytest.mark.parametrize(
    "change", ["single", "dependent", "cancel", "other_reservation", "user_tools"]
)
def test_only_independent_same_reservation_update_targets_are_eligible(change):
    data = inputs()
    if change == "single":
        data["required_actions"].pop()
    elif change == "dependent":
        data["dependencies"] = [["flight", "bags"]]
    elif change == "cancel":
        data["required_actions"].append(
            dict(
                action_id="c",
                name="cancel_reservation",
                arguments={"reservation_id": "R"},
            )
        )
    elif change == "user_tools":
        data["task"]["user_tools"] = ["unverified_user_write"]
    else:
        data["required_actions"][1]["arguments"]["reservation_id"] = "other"
    trace = build([], **data)
    assert not trace["applicable"]
    assert trace["turn_credits"] == [0] * 5


def test_local_credits_preserve_baselines_process_policy_and_observation_masks():
    from test_turn_local_v4 import rows

    group = rows()
    old_config = CreditConfig(version="procredit-turn-v4")
    baseline = compute_group_credit(group, old_config)
    assert "db_subgoal_credit" not in baseline["config"]
    cfg = CreditConfig(
        version="procredit-turn-v4", db_subgoal_credit="db_subgoal_local_v1"
    )
    a = inputs()["required_actions"][0]
    for row in group:
        row["subgoal_trace"] = build([event(a, turn=3)], turns=3, violating_turns=[2])
    updated = compute_group_credit(group, cfg)
    assert updated["turn_mean"] == baseline["turn_mean"]
    assert updated["score_mean"] == baseline["score_mean"]
    assert (
        updated["trajectories"][0]["token_advantages"]
        == baseline["trajectories"][0]["token_advantages"]
    )
    for row in group:
        row["subgoal_trace"] = build([event(a, turn=2)], turns=3, violating_turns=[2])
    updated = compute_group_credit(group, cfg)
    old, new = baseline["trajectories"][0], updated["trajectories"][0]
    assert new["turn_advantages"][1] - old["turn_advantages"][1] == 0.25
    assert new["token_advantages"][2] == 0
    assert new["token_advantages"][-1] <= -1
    with pytest.raises(ValueError):
        CreditConfig(db_subgoal_credit="db_subgoal_local_v1")


def test_missing_snapshots_never_use_old_results_as_current_state():
    a, b = inputs()["required_actions"]
    good = event(a)
    corrupt = event(b, turn=2, sequence=1)
    corrupt.result = "truncated"
    trace = build([good, corrupt])
    assert trace["phi"][2] == 0
    assert trace["turn_credits"] == [0.25, 0, 0, 0, 0]
    assert trace["unknown_state_turns"] == [1]


def test_live_and_frozen_scoring_share_trace_and_reject_tampering(scratch_dir):
    import asyncio

    from test_procredit_rollout import rollout_fixture

    from tau2_agentic_rl.procredit_runtime import credit_from_record
    from tau2_agentic_rl.scoring_retry import retry_scoring

    loop, output, record = rollout_fixture(
        scratch_dir,
        credit_version="procredit-turn-v4",
        db_subgoal_credit=True,
    )
    assert not record.subgoal_trace["applicable"]
    assert output.extra_fields["procredit"]["subgoal_trace"] == record.subgoal_trace
    before = record.subgoal_trace
    scalar = record.custom_reward.train_reward
    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert record.subgoal_trace == before
    assert record.custom_reward.train_reward == scalar
    record.subgoal_trace["turn_credits"][0] = 0.25
    with pytest.raises(ValueError, match="subgoal"):
        credit_from_record(record)


def test_explicit_disabled_switch_is_exactly_the_legacy_audit():
    from test_turn_local_v4 import rows

    cfg = CreditConfig.from_project(
        {"credit": {"version": "procredit-turn-v4", "db_subgoal_credit": None}}
    )
    assert compute_group_credit(rows(), cfg) == compute_group_credit(
        rows(), CreditConfig(version="procredit-turn-v4")
    )


def test_outside_compensation_blocks_bonus_without_erasing_verified_order_state():
    a = inputs()["required_actions"][0]
    compensation = ToolEvent(
        event_id="comp",
        sequence=1,
        turn_id=2,
        name="send_certificate",
        arguments={"user_id": "u", "amount": 100},
        success=True,
        db_effect=True,
        result="certificate sent",
    )
    trace = build([event(a), compensation])
    assert trace["phi"] == [0, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert trace["turn_credits"] == [0.25, 0, 0, 0, 0]
    assert trace["wrong_write_turns"] == [1]


def test_unknown_db_effect_does_not_prove_a_regression():
    a = inputs()["required_actions"][0]
    wrong = deepcopy(a)
    wrong["arguments"]["cabin"] = "business"
    missing_audit = event(wrong, turn=2, sequence=1)
    missing_audit.db_effect = None
    trace = build([event(a), missing_audit])
    assert trace["turn_credits"] == [0.25, 0, 0, 0, 0]
    assert trace["unknown_state_turns"] == [1]


def test_final_snapshot_in_same_turn_controls_bonus_and_complete_itinerary_is_required():
    a = inputs()["required_actions"][0]
    extra = deepcopy(a)
    extra["arguments"]["flights"].append(
        {"flight_number": "WRONG", "date": "2024-05-25"}
    )
    trace = build([event(a), event(extra, sequence=1)])
    assert trace["phi"][1] == 0
    assert not any(v > 0 for v in trace["turn_credits"])
    assert trace["turn_credits"][0] == -0.25


def test_passenger_order_is_irrelevant_but_all_names_and_birthdays_must_match():
    data = inputs()
    first = dict(first_name="A", last_name="B", dob="1970-01-01")
    second = dict(first_name="C", last_name="D", dob="1980-01-01")
    passengers = dict(
        action_id="p",
        name="update_reservation_passengers",
        arguments={"reservation_id": "R", "passengers": [first, second]},
    )
    data["required_actions"].append(passengers)
    e = event(passengers)
    e.arguments["passengers"].reverse()
    result = json.loads(e.result)
    result["passengers"].reverse()
    e.result = json.dumps(result)
    assert build([e], **data)["turn_credits"][0] == pytest.approx(0.5 / 3)
    e.arguments["passengers"][0]["dob"] = "2000-01-01"
    assert build([e], **data)["turn_credits"] == [0] * 5


def test_paid_baggage_requires_both_counts_and_payment():
    b = inputs()["required_actions"][1]
    wrong = event(b)
    wrong.arguments["payment_id"] = "wrong"
    assert build([wrong])["phi"][-1] == 0
    wrong = event(b)
    snapshot = json.loads(wrong.result)
    snapshot["nonfree_baggages"] = 2
    wrong.result = json.dumps(snapshot)
    assert build([wrong])["turn_credits"] == [0] * 5


def test_stale_baggage_state_cannot_waive_a_payment_that_is_actually_due():
    b = inputs()["required_actions"][1]
    wrong = event(b)
    wrong.arguments["payment_id"] = "wrong"
    wrong.state_before = {"reservation_id": "R", "nonfree_baggages": 1}
    assert build([wrong])["turn_credits"] == [0] * 5


@pytest.mark.parametrize("missing", [False, True])
def test_mixed_turn_cannot_certify_safety_from_arguments_with_unverified_result(
    missing,
):
    a, b = inputs()["required_actions"]
    flight = event(a)
    state = json.loads(flight.result)
    state["cabin"] = "business"
    flight.result = "truncated" if missing else json.dumps(state)
    bag = event(b, sequence=1, state=state)
    trace = build([flight, bag])
    assert trace["turn_credits"] == [0] * 5
    assert trace["deferred_turns"] == [0]


def test_enabled_local_credit_reaches_tensors_but_synthetic_padding_stays_zero():
    torch = pytest.importorskip("torch")
    from test_trajectory_reuse import partial_group

    from tau2_agentic_rl.procredit_runtime import build_credit_tensors

    keys, extras, context = partial_group()
    a = inputs()["required_actions"][0]
    for extra in extras:
        extra["procredit"]["subgoal_trace"] = build(
            [event(a, turn=2)], turns=3, violating_turns=[2]
        )
    keys.append("padabc_0_0")
    extras.append(deepcopy(extras[0]))
    mask = torch.tensor([[1, 1, 0, 1, 1]] * 7 + [[0] * 5])
    fields, _ = build_credit_tensors(
        keys,
        extras,
        mask,
        CreditConfig(
            version="procredit-turn-v4", db_subgoal_credit="db_subgoal_local_v1"
        ),
        terminal_groups=context,
        padding=[False] * 7 + [True],
    )
    advantages = fields["advantages"]
    assert advantages[0, 3] > 0
    assert advantages[0, -1] <= -1
    assert torch.count_nonzero(advantages[-1]) == 0
    assert torch.count_nonzero(advantages[:, 2]) == 0
