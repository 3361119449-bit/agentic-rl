"""35-turn truncation is explicit, zero-valued and separate from raw Tau2 facts."""
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.reward.score import (
    RewardConfig,
    build_reward_config,
    score_trajectory,
)
from tau2_agentic_rl.schemas import JudgeResult, OfficialScores

POLICY = "zero-trajectory-v1"
ROOT = Path(__file__).parents[1]
REASONS = ["hard_turn_limit", "budget_exhausted", "context_window_exceeded",
           "generation_truncated", "max_steps"]


def test_current_limits_and_policy_are_explicit():
    for name in ["rl/airline_procredit_v4.yaml", "rl/airline_grpo_v1.yaml",
                 "evaluation/airline_eval_v1.yaml"]:
        cfg = load_yaml(ROOT / "configs" / name)
        assert cfg["rollout"]["max_hard_turns"] == 35
        assert cfg["rollout"]["max_soft_turns"] == 35
        assert cfg["reward"]["truncation_policy"] == POLICY
        assert cfg["reward"]["truncation_multiplier"] == 0
    cfg = load_yaml(ROOT / "configs/rl/airline_procredit_v4.yaml")
    assert cfg["slot_recovery"]["max_truncation_resamples"] == 2
    old = load_yaml(ROOT / "configs/rl/airline_procredit_v4_v11.yaml")
    assert old["rollout"]["max_hard_turns"] == 30
    assert old["rollout"]["max_soft_turns"] == 15
    assert old["reward"]["truncation_multiplier"] == .75


def inputs():
    return dict(events=[], messages=[], assistant_turns=1, required_actions=[],
                official=OfficialScores(reward=1, db_applicable=True, db_score=1),
                judge=JudgeResult(), transfer_rule={})


@pytest.mark.parametrize("reason", REASONS)
def test_zero_scalar_and_strict_keep_raw_official_and_component_facts(reason):
    x = inputs()
    cfg = build_reward_config({"reward": {"truncation_policy": POLICY,
                                         "truncation_multiplier": 0}})
    before = x["official"].model_dump()
    result = score_trajectory(**x, termination_reason=reason, config=cfg)
    assert result.train_reward == result.strict_success == 0
    assert result.components["db"].value == 1
    assert result.details["truncation_zero_applied"] is True
    assert result.details["truncation_multiplier"] == 0
    assert x["official"].model_dump() == before


def test_zero_policy_is_explicit_and_legacy_discount_remains():
    with pytest.raises(ValueError):
        build_reward_config({"reward": {"truncation_policy": "unknown"}})
    with pytest.raises(ValueError):
        build_reward_config({"reward": {"truncation_policy": POLICY,
                                        "truncation_multiplier": .75}})
    legacy = score_trajectory(**inputs(), termination_reason="hard_turn_limit",
                              config=RewardConfig())
    assert legacy.train_reward == .75


@pytest.mark.parametrize("reason", ["user_stop", "agent_stop", "human_transfer"])
def test_natural_completion_keeps_reward_even_when_zero_truncation_is_configured(reason):
    cfg = build_reward_config({"reward": {"truncation_policy": POLICY,
                                         "truncation_multiplier": 0}})
    x = inputs()
    baseline = score_trajectory(**x, termination_reason=reason)
    current = score_trajectory(**x, termination_reason=reason, config=cfg)
    assert current.train_reward == baseline.train_reward == 1
    assert current.strict_success == 1
    assert current.details["truncation_multiplier"] == 1


def group_rows():
    from test_turn_local_v4 import rows
    result = rows()
    for row in result:
        row["score"] = 0
        row["multiplier"] = 0
        row["truncation_policy"] = POLICY
        row["termination_reason"] = "hard_turn_limit"
        row["truncation_zero_applied"] = True
    return result


def test_zero_trajectory_cannot_regain_credit_from_group_costs_or_final_refusal_guard():
    from test_simple_turn_reward import decision, trace
    rows = group_rows()
    for row in rows:
        row["process_credit"]["turn_costs"] = [.1, .2, .3]
        row["simple_turn_labels"] = trace([decision(t) for t in range(3)])
        row["simple_turn_labels"]["turns"][0].update(
            eligible_correct_refusal=True, refusal_bonus=1)
        row["simple_turn_labels"]["turns"][2].update(bad_turn=True)
    result = compute_group_credit(rows, CreditConfig(version="procredit-turn-v4",
                                                    final_turn_rule="simple-turn-v1"))
    assert result["has_signal"] is False
    for row in result["trajectories"]:
        assert set(row["token_advantages"]) == {0}
        assert set(row["turn_advantages"]) == {0}
        assert set(row["turn_returns"]) == {0}
        assert set(row["policy_turn_rewards"]) == {0}
        assert all(x["anew"] == 0 for x in row["simple_turn_audit"])
        assert row["truncation_zero_applied"] is True


@pytest.mark.parametrize("field,value", [("score", 1), ("termination_reason", "user_stop"),
                                        ("truncation_policy", "unknown")])
def test_forged_zero_credit_rows_are_rejected(field, value):
    rows = group_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        compute_group_credit(rows, CreditConfig(version="procredit-turn-v4"))


def test_runtime_zero_credit_needs_no_fabricated_judge_labels(scratch_dir):
    from test_procredit_rollout import rollout_fixture

    from tau2_agentic_rl.procredit_runtime import credit_from_record
    from tau2_agentic_rl.reward.truncation import deterministic_zero_reward
    from tau2_agentic_rl.versions import sha256_json
    _, _, rec = rollout_fixture(scratch_dir, credit_version="procredit-turn-v4")
    rec.termination_reason = "hard_turn_limit"
    rec.scoring_inputs["judge"]["trajectory"]["termination_reason"] = rec.termination_reason
    project = rec.scoring_inputs["reward_project_config"]
    project["reward"].update(truncation_policy=POLICY, truncation_multiplier=0)
    rec.custom_reward = deterministic_zero_reward(
        termination_reason=rec.termination_reason, project=project, official=rec.official_scores)
    rec.judge_result = None
    rec.metadata.pop("simple_turn_labels", None)
    rec.metadata["scoring_inputs_sha256"] = sha256_json(rec.scoring_inputs)
    row = credit_from_record(rec)
    assert row["score"] == row["multiplier"] == 0
    assert row["truncation_zero_applied"] is True
    assert row["truncation_policy"] == POLICY
    group = [deepcopy(row) for _ in range(8)]
    for i, item in enumerate(group):
        item["trajectory_id"] = str(i)
    credit = compute_group_credit(group, CreditConfig(version="procredit-turn-v4"))
    assert not credit["has_signal"]


def test_zero_scoring_retry_does_not_call_remote_judge(scratch_dir):
    from test_review_boundaries import scoring_failure_record

    from tau2_agentic_rl.scoring_retry import retry_scoring
    from tau2_agentic_rl.storage import TrajectoryStore
    from tau2_agentic_rl.versions import sha256_json
    rec = scoring_failure_record()
    rec.termination_reason = "budget_exhausted"
    rec.scoring_inputs["judge"]["trajectory"]["termination_reason"] = rec.termination_reason
    rec.scoring_inputs["reward_project_config"].setdefault("reward", {}).update(
        truncation_policy=POLICY, truncation_multiplier=0)
    rec.metadata["scoring_inputs_sha256"] = sha256_json(rec.scoring_inputs)

    class NoJudge:
        async def evaluate(self, **kwargs):
            raise AssertionError("truncation must not spend a remote Judge call")

    assert asyncio.run(retry_scoring(rec, NoJudge(),
                                     TrajectoryStore(scratch_dir, attach_evaluation_identity=False)))
    assert rec.custom_reward.train_reward == 0
    assert rec.judge_result is None


def test_evaluation_counts_truncation_as_zero_without_replacing_slots(scratch_dir):
    from test_evaluation_summary import setup_evaluation, write_sample

    from scripts.summarize_evaluation import summarize
    from tau2_agentic_rl.evaluation import initialize_evaluation
    root, _, old = setup_evaluation(scratch_dir, count=1)
    # Use a new identity/namespace; historical evaluation identity never mutated.
    manifest = initialize_evaluation(scratch_dir / "zero_eval", {
        **old["identity"], "truncation_policy": POLICY,
        "reward_judge_enabled": False, "user_sim_filter_enabled": False,
    }, resume=False)
    records = scratch_dir / "zero_eval/trajectories"
    records.mkdir()
    original = {}
    for slot in range(4):
        path = write_sample(records, manifest, 0, slot)
        d = json.loads(path.read_text())
        d.update(termination_reason="hard_turn_limit" if slot < 2 else "user_stop",
                 official_scores={"reward": 1}, custom_reward=None)
        path.write_text(json.dumps(d))
        original[path.name] = path.read_bytes()
    result = summarize(records)
    assert result["valid_samples"] == 4
    assert result["missing_slots"] == []
    assert result["aggregate"]["evaluation_pass1"] == .5
    assert result["aggregate"]["evaluation_pass4"] == 0
    assert result["aggregate"]["raw_tau2_official_pass1"] == 1
    assert result["aggregate"]["raw_tau2_official_pass4"] == 1
    assert result["truncated_samples"] == 2
    assert result["metric_definition"] == "tau2_pass_hat_k_zero_truncation_v1"
    assert all(path.read_bytes() == original[path.name] for path in records.glob("*.json"))


def test_native_grpo_zero_mask_runs_after_parent_advantage_and_preserves_other_rows(scratch_dir, monkeypatch):
    from test_procredit_runtime import production_method
    torch = pytest.importorskip("torch")
    td = pytest.importorskip("tensordict")
    runtime_path = scratch_dir / "zero.yaml"
    runtime_path.write_text("reward:\n  truncation_policy: zero-trajectory-v1\n")
    monkeypatch.setenv("AGENTIC_RL_CONFIG", str(runtime_path))
    extras = [
        {"truncation_zero_applied": True, "truncation_policy": POLICY,
         "termination_reason": "hard_turn_limit", "reward_extra_info": {"train_reward": 0}},
        {"truncation_zero_applied": False, "termination_reason": "user_stop"},
    ]
    stored = {"advantages": torch.tensor([[-3., 2.], [1., -1.]]),
              "returns": torch.tensor([[4., -2.], [2., 3.]]),
              "extra_fields": extras, "old_log_probs": torch.tensor([[1., 1.], [2., 2.]])}
    original = stored["old_log_probs"].clone()
    parent_called = []

    class Parent:
        def _compute_advantage(self, batch, metrics):
            parent_called.append(True)
            return batch

    def put(**kwargs):
        stored.update(kwargs["fields"])
        return batch

    cls = production_method("CappedPPOTrainerSync", "_compute_advantage", Parent,
        tq=SimpleNamespace(kv_batch_get=lambda **kw: stored, kv_batch_put=put),
        TensorDict=td.TensorDict)
    trainer = cls()
    trainer._procredit_runtime = lambda: None
    batch = SimpleNamespace(keys=["a_0_0", "a_1_0"], partition_id="train")
    assert trainer._compute_advantage(batch, {}) is batch
    assert parent_called == [True]
    assert stored["advantages"].tolist() == [[0., 0.], [1., -1.]]
    assert stored["returns"].tolist() == [[0., 0.], [2., 3.]]
    assert torch.equal(stored["old_log_probs"], original)


@pytest.mark.parametrize("judge_enabled", [True, False])
def test_actor_35th_unfinished_turn_records_cutoff_raw_score_and_zero_without_judge(scratch_dir, judge_enabled):
    from test_procredit_rollout import rollout_fixture
    loop, output, rec = rollout_fixture(scratch_dir, credit_version="procredit-turn-v4" if judge_enabled else None,
        zero_truncation=True, hard_limit=35, judge_failures=100, judge_enabled=judge_enabled)
    assert rec.assistant_turns == output.num_turns == 35
    assert rec.termination_reason == "hard_turn_limit"
    assert rec.metadata["truncation"]["occurred"] is True
    assert rec.metadata["truncation"]["will_resample"] is True
    assert rec.metadata["truncation"]["resample_limit"] == 2
    assert rec.official_scores.reward == 1
    assert output.reward_score == 0
    assert output.extra_fields["truncation_zero_applied"] is True
    assert rec.judge_result is None
    assert loop.test_judge_calls == []
    if judge_enabled:
        assert rec.custom_reward.train_reward == rec.custom_reward.strict_success == 0
        assert rec.progress_trace["phi"][-1] == 1
        assert set(output.extra_fields["procredit"]["phi"]) == {0}
    else:
        assert rec.custom_reward is None


def test_natural_completion_on35_is_not_truncated(scratch_dir):
    from test_procredit_rollout import rollout_fixture
    _, output, rec = rollout_fixture(scratch_dir, credit_version="procredit-turn-v4",
        zero_truncation=True, hard_limit=35, natural_stop_at=35)
    assert rec.assistant_turns == 35
    assert rec.termination_reason != "hard_turn_limit"
    assert rec.metadata["truncation"]["occurred"] is False
    assert rec.custom_reward.train_reward > 0
    assert output.extra_fields["truncation_zero_applied"] is False


@pytest.mark.parametrize("reason", ["max_steps", "context_window_exceeded", "timeout"])
def test_prior_transfer_cannot_hide_actual_environment_cutoff(reason):
    import ast

    from tau2_agentic_rl.schemas import ToolEvent
    source = (ROOT / "src/tau2_agentic_rl/agent_loop/airline.py").read_text()
    tree = ast.parse(source)
    mapping = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_termination_reason")
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == "terminated")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), mapping, branch], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"terminated": True, "environment": SimpleNamespace(
        tau2_termination_reason=lambda: reason), "tool_events": [
            ToolEvent(event_id="transfer", sequence=0, turn_id=1,
                      name="transfer_to_human_agents", success=True)]}
    exec(compile(module, "<actual actor termination branch>", "exec"), scope)
    assert scope["termination_reason"] == reason
