import asyncio
import sys
from pathlib import Path

import pytest
import yaml
from test_procredit_rollout import rollout_fixture

from scripts import rescore_saved_trajectories as script
from tau2_agentic_rl.agent_policy import load_agent_system_prompt, prompt_sha256
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.judge.prompts import rubric_fingerprint
from tau2_agentic_rl.procredit_runtime import credit_from_record
from tau2_agentic_rl.scoring_retry import retry_scoring
from tau2_agentic_rl.storage import TrajectoryStore


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_offline_changed_cost_remains_replayable_with_new_frozen_config(scratch_dir, monkeypatch, version):
    root = Path(__file__).parents[1]
    loop, _, record = rollout_fixture(scratch_dir, credit_version=f"procredit-turn-{version}")
    cfg = load_yaml(root / f"configs/rl/airline_procredit_{version}.yaml")
    cfg["reward"]["process_penalties"]["mixed_content_and_tool_call"] = .05
    config_path = scratch_dir / "changed.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    record.metadata["agent_system_prompt_sha256"] = prompt_sha256(load_agent_system_prompt(cfg, root))
    record.metadata["judge_rubric_sha256"] = rubric_fingerprint(
        [], [], {}, require_policy_attribution=version == "v4",
    )
    monkeypatch.setattr(script, "load_required_actions", lambda path: {"0": []})
    monkeypatch.setattr(script, "load_action_dependencies", lambda path: {})
    monkeypatch.setattr(script, "load_task_mapping", lambda path: {"0": {}})
    source, output = scratch_dir / "source", scratch_dir / "rescored"
    TrajectoryStore(source, attach_evaluation_identity=False).save(record)
    original = (source / f"{record.trajectory_id}.json").read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "rescore", str(source), str(output), "--config", str(config_path),
        "--project-root", str(root), "--reward-version", "v3-rescored",
    ])
    script.main()
    store = TrajectoryStore(output, attach_evaluation_identity=False)
    rescored = next(store.records())
    row = credit_from_record(rescored)
    assert row["process_credit"]["turn_costs"] == pytest.approx([0, .05, 0])
    expected = rescored.custom_reward.model_dump()
    rescored.custom_reward = None
    rescored.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(rescored, loop.judge, store))
    assert rescored.custom_reward.model_dump() == expected
    assert (source / f"{record.trajectory_id}.json").read_bytes() == original
