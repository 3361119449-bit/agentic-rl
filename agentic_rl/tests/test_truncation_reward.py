import asyncio
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_review_boundaries import scoring_failure_record

from tau2_agentic_rl.agent_policy import load_agent_system_prompt, prompt_sha256
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.judge.prompts import rubric_fingerprint
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, OfficialScores, ToolEvent
from tau2_agentic_rl.scoring_retry import retry_scoring
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.versions import sha256_json


def reward_inputs(transfer=False):
    return dict(
        events=(
            [
                ToolEvent(
                    event_id="transfer",
                    sequence=0,
                    turn_id=3,
                    name="transfer_to_human_agents",
                    success=True,
                )
            ]
            if transfer
            else []
        ),
        messages=[],
        assistant_turns=16,  # A real process penalty must be applied BEFORE ×0.75.
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
        transfer_rule={"allowed": True} if transfer else {},
    )


@pytest.mark.parametrize("transfer", [False, True], ids=["normal", "transfer"])
@pytest.mark.parametrize(
    "reason",
    [
        "generation_truncated",
        "budget_exhausted",
        "hard_turn_limit",
        "max_steps",
        "context_window_exceeded",
    ],
)
def test_final_reward_discount_is_once_and_does_not_change_components(transfer, reason):
    inputs = reward_inputs(transfer)
    before = inputs["official"].model_dump()
    baseline = score_trajectory(**inputs, termination_reason="user_stop")
    result = score_trajectory(**inputs, termination_reason=reason)
    assert result.branch == ("human_transfer" if transfer else "normal")
    assert baseline.process_penalty == pytest.approx(0.02)
    assert baseline.train_reward == pytest.approx(0.98)
    assert result.train_reward == pytest.approx(0.98 * 0.75)
    assert result.strict_success == baseline.strict_success == 1
    assert result.progress == baseline.progress
    assert result.components == baseline.components
    assert result.process_penalty == baseline.process_penalty
    assert result.details["trajectory_truncated"]
    assert result.details["reward_before_truncation"] == baseline.train_reward
    assert inputs["official"].model_dump() == before


@pytest.mark.parametrize(
    "reason",
    [
        "user_stop",
        "agent_stop",
        "human_transfer",
        "environment_terminated",
        "timeout",
    ],
)
def test_nontruncated_termination_and_clipped_observations_are_not_discounted(reason):
    inputs = reward_inputs()
    inputs["events"] = [
        ToolEvent(
            event_id="read",
            sequence=0,
            turn_id=1,
            name="get_user_details",
            success=True,
            observation_truncated=True,
        )
    ]
    result = score_trajectory(**inputs, termination_reason=reason)
    assert result.train_reward == pytest.approx(0.98)
    assert result.details["truncation_multiplier"] == 1
    assert not result.details["trajectory_truncated"]


@pytest.mark.parametrize("transfer", [False, True])
def test_gate_zero_remains_zero_after_discount(transfer):
    inputs = reward_inputs(transfer)
    inputs["judge"] = JudgeResult(
        mandatory_policy_checks=[JudgeCheck(criterion_id="policy", passed=False)]
    )
    result = score_trajectory(**inputs, termination_reason="hard_turn_limit")
    assert result.train_reward == result.details["reward_before_truncation"] == 0
    assert result.strict_success == 0


def test_yaml_wires_discount_into_both_runtime_profiles():
    root = Path(__file__).parents[1]
    for config in ["rl/airline_grpo_v1.yaml", "evaluation/airline_eval_v1.yaml"]:
        project = load_yaml(root / "configs" / config)
        assert project["project"]["reward_version"] == "v3-truncation-evidence"
        assert build_reward_config(project).truncation_multiplier == 0.75
    configured = score_trajectory(
        **reward_inputs(),
        termination_reason="hard_turn_limit",
        config=build_reward_config({"reward": {"truncation_multiplier": 0.5}}),
    )
    assert configured.train_reward == pytest.approx(0.98 * 0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_bad_discount_is_rejected(value):
    with pytest.raises(ValueError, match="truncation_multiplier"):
        build_reward_config({"reward": {"truncation_multiplier": value}})


def test_scoring_retry_preserves_truncation_and_applies_discount_once(scratch_dir):
    record = scoring_failure_record()
    record.termination_reason = "generation_truncated"
    record.scoring_inputs["judge"]["trajectory"]["termination_reason"] = (
        record.termination_reason
    )
    record.official_scores = OfficialScores(reward=0, db_applicable=True, db_score=1)
    record.scoring_inputs["official_scores"] = record.official_scores.model_dump()
    record.metadata["scoring_inputs_sha256"] = sha256_json(record.scoring_inputs)
    before = deepcopy(record.scoring_inputs)

    class Judge:
        async def evaluate(self, **kwargs):
            return JudgeResult(), "raw", "prompt", "cache"

    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    assert asyncio.run(retry_scoring(record, Judge(), store))
    saved = next(store.records())
    assert saved.custom_reward.train_reward == 0.75
    assert saved.scoring_inputs == before
    assert saved.official_scores.reward == 0
    with pytest.raises(ValueError, match="not a scoring-only failure"):
        asyncio.run(retry_scoring(record, Judge(), store))


def test_bad_retry_evidence_stays_pending_instead_of_becoming_a_zero_score(scratch_dir):
    record = scoring_failure_record()

    class Judge:
        async def evaluate(self, **kwargs):
            return (
                JudgeResult(
                    semantic_checks=[
                        JudgeCheck(
                            criterion_id="s",
                            passed=True,
                            evidence_turn_ids=[99],
                        )
                    ]
                ),
                "raw",
                "prompt",
                "cache",
            )

    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    assert not asyncio.run(retry_scoring(record, Judge(), store))
    saved = next(store.records())
    assert saved.custom_reward is None
    assert saved.official_scores.reward == 1
    assert saved.metadata["failure_phase"] == "reward_scoring"
    assert "absent turn IDs" in saved.metadata["failure_message"]


@pytest.mark.parametrize("invalid_evidence", [False, True])
def test_offline_rescore_uses_saved_termination_and_validates_evidence(
    scratch_dir, monkeypatch, invalid_evidence
):
    from scripts import rescore_saved_trajectories as script

    root = Path(__file__).parents[1]
    config_path = root / "configs/rl/airline_grpo_v1.yaml"
    config = load_yaml(config_path)
    record = scoring_failure_record()
    record.termination_reason = "budget_exhausted"
    record.official_scores = OfficialScores(reward=0, db_applicable=True, db_score=1)
    record.judge_result = JudgeResult(
        semantic_checks=[
            JudgeCheck(
                criterion_id="s",
                passed=True,
                evidence_turn_ids=[99 if invalid_evidence else 0],
            )
        ]
    )
    record.environment_transcript[0]["turn_idx"] = 0
    record.metadata["agent_system_prompt_sha256"] = prompt_sha256(
        load_agent_system_prompt(config, root)
    )
    record.metadata["judge_rubric_sha256"] = rubric_fingerprint([], [], {})
    # Isolate the script's scorer wiring from task annotation content.
    monkeypatch.setattr(script, "load_required_actions", lambda path: {"0": []})
    monkeypatch.setattr(script, "load_action_dependencies", lambda path: {})
    monkeypatch.setattr(script, "load_task_mapping", lambda path: {"0": {}})
    source = scratch_dir / "source"
    TrajectoryStore(source, attach_evaluation_identity=False).save(record)
    source_bytes = (source / "same-id.json").read_bytes()

    def run_rescore(input_dir, output_dir):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rescore",
                str(input_dir),
                str(output_dir),
                "--config",
                str(config_path),
                "--project-root",
                str(root),
                "--reward-version",
                "v3-truncation-evidence",
            ],
        )
        script.main()

    first = scratch_dir / "first"
    if invalid_evidence:
        with pytest.raises(ValueError, match="absent turn IDs"):
            run_rescore(source, first)
        assert not list(first.glob("*.json"))
    else:
        run_rescore(source, first)
        second = scratch_dir / "second"
        run_rescore(first, second)
        for output in [first, second]:
            result = next(
                TrajectoryStore(output, attach_evaluation_identity=False).records()
            )
            assert result.custom_reward.train_reward == 0.75  # Never 0.75 squared.
            assert result.official_scores.reward == 0
            assert (
                result.custom_reward.details["termination_reason"] == "budget_exhausted"
            )
    assert (source / "same-id.json").read_bytes() == source_bytes
