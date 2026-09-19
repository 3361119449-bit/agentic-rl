"""Bound multitool SFT prompt -> actor/Judge identity and policy contracts."""

import json
import sys
from pathlib import Path

import pytest

from scripts.build_annotations import _policy_rows
from scripts.export_sft_system_prompt import extract
from tau2_agentic_rl.agent_policy import (
    extract_airline_policy,
    load_agent_system_prompt,
    prompt_sha256,
)
from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.initial_prompt import initial_messages
from tau2_agentic_rl.judge.prompts import JUDGE_SYSTEM_PROMPT, build_judge_messages
from tau2_agentic_rl.policy_rules import policy_checks, validate_policy_rows
from tau2_agentic_rl.reward.process_penalty import compute_process_penalty
from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, OfficialScores, ToolEvent

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/rl/airline_grpo_v1.yaml"
DIGEST = "e490e0859e1357ed6dd777fdbddab15b11d6de96ff7161328a6babcab77348ab"


def test_actor_and_evaluation_use_exact_sft_message_without_extra_protocol():
    project = load_yaml(CONFIG)
    prompt = load_agent_system_prompt(project, ROOT)
    evaluation = load_yaml(ROOT / "configs/evaluation/airline_eval_v1.yaml")
    assert load_agent_system_prompt(evaluation, ROOT) == prompt
    assert prompt_sha256(prompt) == DIGEST
    incoming = [{"role": "user", "content": "Cancel my reservation."}]
    assert initial_messages(prompt, incoming) == [
        {"role": "system", "content": prompt},
        *incoming,
    ]
    assert "action_proposal" not in prompt
    assert "Make one or more tool calls." in prompt
    assert "You may make one or more tool calls in the same turn." in prompt
    assert "obtain explicit user confirmation (yes) to proceed" in prompt


def test_frozen_prompt_artifact_matches_bound_uploaded_prompt_identity():
    artifact = json.loads(
        (ROOT / "configs/prompts/areal_airline_sft.v1.json").read_text(encoding="utf-8")
    )
    assert artifact["system_prompt_sha256"] == DIGEST
    assert prompt_sha256(artifact["system_prompt"]) == DIGEST
    assert artifact["source_rows"] == 8
    assert artifact["source_sha256"] == (
        "b1f6c296033283a3c4550b8c2b9cc7dc5f1bb1f0e00e2236a22e2cd6d3e9fa1d"
    )


def test_export_rejects_mixed_prompts_instead_of_choosing_first(scratch_dir):
    source = scratch_dir / "input.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "system", "content": text}]})
            for text in ("one", "two")
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple SFT system prompts"):
        extract(source)


def test_missing_or_modified_prompt_never_falls_back_to_official(scratch_dir):
    with pytest.raises(ValueError, match="no official-policy fallback"):
        load_agent_system_prompt({}, scratch_dir)
    config = load_yaml(CONFIG)
    artifact = json.loads(
        (ROOT / config["agent_policy"]["path"]).read_text(encoding="utf-8")
    )
    artifact["system_prompt"] += " changed"
    (scratch_dir / "prompt.json").write_text(json.dumps(artifact), encoding="utf-8")
    config["agent_policy"]["path"] = "prompt.json"
    with pytest.raises(ValueError, match="hash mismatch"):
        load_agent_system_prompt(config, scratch_dir)


def test_environment_cannot_inject_another_system_message():
    with pytest.raises(ValueError, match="environment system override"):
        initial_messages("SFT", [{"role": "system", "content": "official"}])


@pytest.mark.parametrize("split", ["train", "test"])
def test_checked_in_policy_bundles_match_multitool_generator(split):
    mapping = load_task_mapping(
        ROOT / f"data/annotations/airline_mandatory_policy_rules.{split}.v1.json"
    )
    validate_policy_rows(mapping)
    assert list(mapping.values()) == _policy_rows(set(mapping))
    serialized = json.dumps(mapping)
    assert "confirmation_details" not in serialized
    assert "confirmation_before_database_write" not in serialized
    assert "database_write_confirmation" in serialized
    assert "one_tool_call_per_assistant_turn" not in serialized
    mapping[next(iter(mapping))]["judge_checks"].append(
        {"criterion_id": "confirmation_details"}
    )
    with pytest.raises(ValueError, match="stale policy rubric"):
        validate_policy_rows(mapping)


def test_judge_uses_bound_multitool_policy_and_enforces_confirmation():
    prompt = load_agent_system_prompt(load_yaml(CONFIG), ROOT)
    policy = extract_airline_policy(prompt)
    messages = build_judge_messages(
        task={},
        policy=policy,
        trajectory={},
        semantic_checks=[],
        mandatory_policy_checks=policy_checks("0"),
        transfer_rule={},
    )
    assert policy in messages[1]["content"]
    assert (
        "In each turn you can either" not in policy
    )  # Agent-output instructions aren't inside the policy block.
    assert "database-write confirmation exactly as stated" in JUDGE_SYSTEM_PROMPT
    assert "confirmation_details" not in messages[1]["content"]
    assert "database_write_confirmation" in messages[1]["content"]
    assert "travel_certificate_new_booking_only" not in messages[1]["content"]
    result = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(
            mandatory_policy_checks=[
                JudgeCheck(criterion_id="0:policy:information_grounding", passed=False)
            ]
        ),
    )
    assert not result.policy_gate and result.train_reward == 0


def test_legacy_confirmation_error_does_not_create_process_penalty():
    event = ToolEvent(
        event_id="legacy",
        sequence=0,
        turn_id=1,
        name="cancel_reservation",
        error_kind="confirmation_required",
        unchanged_retry=True,
    )
    assert compute_process_penalty([event], 1).penalty == 0


def test_offline_rescore_rejects_old_actor_policy_before_writing_scores(
    scratch_dir, monkeypatch
):
    from scripts import rescore_saved_trajectories as script
    from tau2_agentic_rl.schemas import TrajectoryRecord

    source, output = scratch_dir / "old", scratch_dir / "new"
    source.mkdir()
    record = TrajectoryRecord(
        trajectory_id="old",
        task_id="0",
        split="train",
        termination_reason="user_stop",
        annotation_version="old",
        reward_version="old",
        policy_version=0,
        assistant_turns=1,
        trajectory_tokens=10,
    )
    (source / "old.json").write_text(record.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescore",
            str(source),
            str(output),
            "--config",
            str(CONFIG),
            "--project-root",
            str(ROOT),
        ],
    )
    with pytest.raises(ValueError, match="bound SFT system prompt"):
        script.main()
    assert not list(output.glob("*.json"))
