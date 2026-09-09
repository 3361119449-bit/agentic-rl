"""SFT -> actor/Judge identity and the absence of a hidden confirmation gate."""

import json
import sys
from pathlib import Path

import pytest

from scripts.build_annotations import _policy_rows
from scripts.export_sft_system_prompt import DEFAULT_SOURCE, extract
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
DIGEST = "121cad8bac61dc66ff58177383492b736380d9006aee4c5249e08f1315eef7d9"


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
    assert "It is recommended" in prompt and "LLM JUDGE SHOULD NOT CARE" in prompt


def test_frozen_prompt_matches_every_source_sft_row_when_lfs_is_available():
    with DEFAULT_SOURCE.open("rb") as handle:
        if handle.read(100).startswith(b"version https://git-lfs.github.com/spec"):
            pytest.skip("full SFT dataset is an unfetched LFS object")
    artifact = extract(DEFAULT_SOURCE)
    expected = json.loads(
        (ROOT / "configs/prompts/areal_airline_sft.v1.json").read_text(encoding="utf-8")
    )
    assert artifact == expected
    assert artifact["source_rows"] == 10649


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
def test_checked_in_policy_bundles_match_generator_without_confirmation(split):
    mapping = load_task_mapping(
        ROOT / f"data/annotations/airline_mandatory_policy_rules.{split}.v1.json"
    )
    validate_policy_rows(mapping)
    assert list(mapping.values()) == _policy_rows(set(mapping))
    assert "confirmation_details" not in json.dumps(mapping)
    assert "confirmation_before_database_write" not in json.dumps(mapping)
    mapping[next(iter(mapping))]["judge_checks"].append(
        {"criterion_id": "confirmation_details"}
    )
    with pytest.raises(ValueError, match="stale policy rubric"):
        validate_policy_rows(mapping)


def test_judge_uses_sft_policy_and_exempts_confirmation_but_not_other_checks():
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
        "CRITICAL RULES" not in policy
    )  # Agent-output instructions aren't Judge rules.
    assert "Do not fail ANY criterion solely for missing" in JUDGE_SYSTEM_PROMPT
    assert "confirmation_details" not in messages[1]["content"]
    assert "travel_certificate_new_booking_only" in messages[1]["content"]
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
