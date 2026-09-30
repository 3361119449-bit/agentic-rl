from copy import deepcopy

import pytest
from test_policy_credit import policy_row
from test_procredit_rollout import rollout_fixture

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.procredit_runtime import credit_from_record


def rows():
    result = [policy_row(i, score=0, phi=[0, 0, 0, 0]) for i in range(8)]
    for row in result:
        row["credit_version"] = "procredit-turn-v4"
        row["process_credit"] = {"version": "process-credit-v2", "turn_costs": [0, 0, 0]}
    return result


def test_violation_tokens_stay_negative_even_with_best_trajectory_reward():
    group = rows()
    group[0].update(score=1.5, terminal_success=1, phi=[0, .5, 1, 1])
    result = compute_group_credit(group, CreditConfig(version="procredit-turn-v4"))
    best = result["trajectories"][0]
    assert best["trajectory_advantage"] > 1
    assert best["token_advantages"][-1] <= -1
    assert best["token_advantages"][0] > 0
    assert best["token_advantages"][2] == 0  # Observation token.
    assert best["policy_turn_rewards"] == [0, 0, -1]


def test_all_violating_group_is_kept_and_every_bad_turn_is_negative():
    result = compute_group_credit(rows(), CreditConfig(version="procredit-turn-v4"))
    assert result["has_signal"]
    assert all(item["token_advantages"][-1] <= -1 for item in result["trajectories"])


def test_v4_cannot_disable_policy_punishment_or_train_unlocalized_violations():
    with pytest.raises(ValueError):
        CreditConfig(version="procredit-turn-v4", turn_coefficient=0)
    group = rows()
    group[0]["policy_credit"].update(attribution_complete=False, unresolved_checks=["p"])
    with pytest.raises(ValueError, match="attribution"):
        compute_group_credit(group, CreditConfig(version="procredit-turn-v4"))
    group = rows()
    group[0]["response_turn_ids"] = [0, 0, -1, 1, 1]
    with pytest.raises(ValueError, match="tokens"):
        compute_group_credit(group, CreditConfig(version="procredit-turn-v4"))


def test_v4_actor_preserves_score_on_policy_failure_and_freezes_local_rewards(scratch_dir):
    _, output, record = rollout_fixture(scratch_dir, credit_version="procredit-turn-v4")
    assert not record.custom_reward.policy_gate
    assert record.custom_reward.strict_success == 0  # Audit metric remains strict.
    assert record.custom_reward.train_reward > 0
    assert record.custom_reward.train_reward == pytest.approx(
        record.custom_reward.details["truncation_multiplier"] * (
            record.custom_reward.details["task_completion"] + .5 * record.progress_trace["phi"][-1]
        )
    )
    assert record.custom_reward.process_penalty == 0
    assert record.custom_reward.details["policy_turn_rewards"] == [0, -1, 0]
    row = credit_from_record(record)
    assert row["score"] == output.reward_score
    assert row["terminal_success"] == 1  # Ungated task completion used for learning.
    group = []
    for i in range(8):
        item = deepcopy(row)
        item["trajectory_id"] = f"sample{i}"
        group.append(item)
    result = compute_group_credit(group, CreditConfig(version="procredit-turn-v4"))
    assert result["trajectories"][0]["token_advantages"][3] <= -1
    with pytest.raises(ValueError):
        compute_group_credit(group, CreditConfig(version="procredit-turn-v3"))


def test_v4_judge_failure_retries_only_the_frozen_scoring_input(scratch_dir):
    _, output, record = rollout_fixture(
        scratch_dir, credit_version="procredit-turn-v4", judge_failures=1,
    )
    assert record.metadata["scoring_retries"][-1]["success"]
    assert len(record.token_turns) == 3
    assert output.response_ids == [4, 9, 7, 5, 9, 7, 6]
    assert len(list(scratch_dir.rglob(record.trajectory_id + ".json"))) == 1
    assert record.custom_reward.train_reward > 0


def test_v4_persistent_scoring_failure_stops_without_a_new_interaction(scratch_dir):
    from tau2_agentic_rl.slot_recovery import SlotRecoveryExhausted
    from tau2_agentic_rl.storage import TrajectoryStore

    with pytest.raises(SlotRecoveryExhausted, match="frozen trajectory"):
        rollout_fixture(scratch_dir, credit_version="procredit-turn-v4", judge_failures=100)
    paths = list(scratch_dir.rglob("*.json"))
    assert len(paths) == 1
    record = next(TrajectoryStore(paths[0].parent, attach_evaluation_identity=False).records())
    assert len(record.token_turns) == 3
    assert len(record.metadata["scoring_retries"]) == 2
    assert record.custom_reward is None


def test_v4_negative_violation_advantage_reaches_actual_trainer_with_real_tensordict():
    from types import SimpleNamespace

    from test_procredit_runtime import production_method, queue_rows

    torch = pytest.importorskip("torch")
    td = pytest.importorskip("tensordict")
    keys, _ = queue_rows()
    group = rows()
    group[0].update(score=1.5, terminal_success=1, phi=[0, .5, 1, 1])
    group[0]["process_credit"]["turn_costs"] = [0, 0, .08]
    extras = [{"procredit": row} for row in group]
    old = torch.randn(8, 7)
    original = old.clone()
    stored = td.TensorDict({
        "response_mask": torch.tensor([[1, 1, 0, 1, 1, 0, 0]] * 8),
        "extra_fields": td.NonTensorStack(*[td.NonTensorData(e) for e in extras]),
        "old_log_probs": old,
    }, batch_size=8)

    def put(**kwargs):
        stored.update(kwargs["fields"])
        return batch

    class Parent:
        def _compute_advantage(self, *args, **kwargs):
            raise AssertionError("scalar GRPO must not overwrite local punishment")

    cls = production_method(
        "CappedPPOTrainerSync", "_compute_advantage", Parent,
        tq=SimpleNamespace(kv_batch_get=lambda **kw: stored, kv_batch_put=put),
        TensorDict=td.TensorDict,
    )
    trainer = cls()
    trainer._procredit_runtime = lambda: (CreditConfig(version="procredit-turn-v4"), None)
    batch = SimpleNamespace(keys=keys, partition_id="train")
    trainer._compute_advantage(batch, {})
    assert torch.all(stored["advantages"][:, 4] <= -1)
    assert stored["advantages"][0, 4] <= -1.08
    assert stored["advantages"][0, 0] > 0
    assert torch.all(stored["advantages"][:, [2, 5, 6]] == 0)
    assert torch.equal(stored["old_log_probs"], original)


def test_v4_policy_does_not_clip_progress_or_erase_any_task_return():
    invalid = rows()
    for row in invalid:
        row.update(score=1.5, terminal_success=1, phi=[0, 0, 0, 1])
    valid = deepcopy(invalid)
    for row in valid:
        row["valid"] = True
        row["policy_credit"]["violating_turns"] = []
    cfg = CreditConfig(version="procredit-turn-v4")
    bad = compute_group_credit(invalid, cfg)
    good = compute_group_credit(valid, cfg)
    assert bad["turn_mean"] == good["turn_mean"]
    assert bad["trajectories"][0]["turn_returns"] == [1.5, 1.5, 1.5]
    assert bad["trajectories"][0]["token_advantages"][:4] == good["trajectories"][0]["token_advantages"][:4]
    assert bad["trajectories"][0]["token_advantages"][-1] == -1


def test_v4_process_is_additive_local_and_keeps_fixed_cost_above_old_total_cap():
    group = rows()
    for row in group:
        row.update(valid=True, phi=[0, 0, 0, 0])
        row["policy_credit"]["violating_turns"] = []
        row["process_credit"]["turn_costs"] = [.3, .08, .06]
    cfg = CreditConfig(version="procredit-turn-v4")
    clean = deepcopy(group)
    for row in clean:
        row["process_credit"]["turn_costs"] = [0, 0, 0]
    a, b = compute_group_credit(group, cfg), compute_group_credit(clean, cfg)
    assert a["score_mean"] == b["score_mean"]
    assert a["turn_mean"] == b["turn_mean"]
    assert a["trajectories"][0]["turn_advantages"] == pytest.approx([-.3, -.08, -.06])
    for row in group:
        row["valid"] = False
        row["policy_credit"]["violating_turns"] = [1]
    bad = compute_group_credit(group, cfg)["trajectories"][0]
    assert bad["turn_advantages"] == pytest.approx([-.3, -1.08, -.06])
    group[0].update(score=1.5, terminal_success=1, phi=[0, .5, 1, 1])
    high = compute_group_credit(group, cfg)["trajectories"][0]
    assert high["token_advantages"][3] <= -1.08


def test_v4_process_subtracts_fixed_cost_without_clipping_positive_advantage():
    group = rows()
    for row in group:
        row.update(valid=True, phi=[0, 0, 0, 0])
        row["policy_credit"]["violating_turns"] = []
    group[0].update(score=.5, phi=[0, 0, 0, 1])
    cfg = CreditConfig(version="procredit-turn-v4")
    before = compute_group_credit(group, cfg)
    group[0]["process_credit"]["turn_costs"] = [0, .08, 0]
    after = compute_group_credit(group, cfg)
    assert after["trajectories"][0]["token_advantages"][3] == pytest.approx(
        before["trajectories"][0]["token_advantages"][3] - .08
    )
    for t in [0, 1, 2, 4]:
        assert after["trajectories"][0]["token_advantages"][t] == before["trajectories"][0]["token_advantages"][t]


def test_v4_many_process_errors_and_extra_turns_never_dilute_fixed_costs():
    from tau2_agentic_rl.reward.process_penalty import (
        ProcessPenaltyConfig,
        build_process_credit,
    )
    from tau2_agentic_rl.schemas import ToolEvent

    events = [ToolEvent(event_id=f"e{i}", sequence=i, turn_id=i+1,
                        error_kind="model_caused_execution_error", unchanged_retry=True,
                        no_progress=True) for i in range(20)]
    trace = build_process_credit(
        events, 20, ProcessPenaltyConfig(cap=None, over_turn_cap=None), local_only=True,
    )
    assert trace["version"] == "process-credit-v2"
    assert trace["turn_costs"][:15] == pytest.approx([.17] * 15)
    assert trace["turn_costs"][15:] == pytest.approx([.19] * 5)
    assert trace["total_cost"] == pytest.approx(3.5)


@pytest.mark.parametrize("allowed,judge_valid,completed", [(True, False, 1), (False, True, 0), (False, True, 1)])
def test_v4_transfer_policy_failure_does_not_gate_component_completion(allowed, judge_valid, completed):
    from types import SimpleNamespace

    from tau2_agentic_rl.reward.progress import build_progress_trace
    from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
    from tau2_agentic_rl.schemas import (
        JudgeResult,
        OfficialScores,
        ToolEvent,
        TransferCheck,
    )

    cfg = {"reward": {"mode": "turn_local_v1", "mandatory_policy_gate": False,
                      "task_safety_gate": False}, "credit": {"version": "procredit-turn-v4"}}
    rule = {"allowed": allowed, "required": allowed}
    events = [ToolEvent(event_id="transfer", sequence=0, turn_id=2,
                        name="transfer_to_human_agents", success=True)]
    judge = JudgeResult(transfer_check=TransferCheck(applicable=True, valid=judge_valid))
    trace = build_progress_trace(
        task={"id": "0"}, messages=[{}] * 4, prefix_lengths=[1, 2, 3, 4],
        events=events, required_actions=[], dependencies=[], transfer_rule=rule,
        initial_state_fingerprint="initial", version="progress-v2",
        evaluator=lambda task, messages: {"db": None, "communicate": []},
    )
    reward = score_trajectory(
        events=events, messages=[], assistant_turns=3, required_actions=[],
        official=OfficialScores(reward=completed, db_applicable=True, db_score=completed),
        judge=judge, transfer_rule=rule,
        config=build_reward_config(cfg), progress_trace=trace,
    )
    assert not reward.policy_gate and reward.strict_success == 0
    assert reward.branch == ("human_transfer" if allowed else "normal")
    assert reward.details["task_completion"] == completed
    assert reward.train_reward == completed + .5 * trace["phi"][-1]
    assert reward.details["policy_turn_rewards"] == [0, -1, 0]
    record = SimpleNamespace(
        schema_version="2.0", custom_reward=reward, progress_trace=trace,
        response_turn_ids=[0, 1, 2], assistant_turns=3, trajectory_tokens=4,
        token_turns=[SimpleNamespace(
            prompt_token_ids=list(range(i + 1)), output_token_ids=[9], assistant_turn_index=i,
        ) for i in range(3)],
        trajectory_id="transfer", task_id="0", policy_version=0,
        judge_result=judge, tool_events=events, scoring_inputs={
            "reward_project_config": cfg, "judge": {"transfer_rule": rule},
        },
    )
    assert credit_from_record(record)["terminal_success"] == completed
