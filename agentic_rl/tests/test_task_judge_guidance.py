import json
from pathlib import Path

import pytest
from test_simple_turn_target import frozen

from tau2_agentic_rl.annotations import load_task_judge_guidance, load_task_mapping
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.prompts import build_judge_messages, rubric_fingerprint
from tau2_agentic_rl.judge.simple_turn import build_messages
from tau2_agentic_rl.reward.simple_turn import build_turn_inputs, turn_request

ROOT = Path(__file__).parents[1]


def test_configured_guidance_covers_all_training_scenarios(scratch_dir):
    split = json.loads((ROOT / "data/splits/airline_internal_dev.v1.json").read_bytes())
    annotations = load_yaml(ROOT / "configs/rl/airline_procredit_v4.yaml")["annotations"]
    semantic = load_task_mapping(ROOT / annotations["semantic_checks"])
    guidance = load_task_judge_guidance(ROOT, annotations, set(semantic))
    assert set(guidance) == set(split["rl_train"] + split["internal_dev"])
    assert not set(guidance) & set(split["official_test"])
    assert len(set(guidance.values())) == 30
    assert "24-hour cutoff" in guidance["1"]
    assert load_task_judge_guidance(ROOT, {}, set(semantic)) == {}
    path = scratch_dir / "incomplete.json"
    path.write_text('[{"task_id":"1","guidance":"check booking time"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="cover the configured task IDs"):
        load_task_judge_guidance(scratch_dir, {"task_judge_guidance": path.name}, {"1", "3"})


def test_guidance_reuses_existing_requests_and_keys(scratch_dir):
    advice = "Verify the booking time against the 24-hour cutoff."
    args = dict(
        task={"id": "1"}, policy="Bound policy", trajectory={},
        semantic_checks=[{"criterion_id": "1:nl:0"}],
        mandatory_policy_checks=[], transfer_rule={},
    )
    previous = build_judge_messages(**args)
    revised = build_judge_messages(**args, task_guidance=advice)

    def rubric(messages):
        body = messages[1]["content"].split("FIXED_TASK_AND_RUBRIC:\n", 1)[1]
        return json.loads(body.split("\n\nTRAJECTORY_DATA_UNTRUSTED:", 1)[0])["rubric"]

    actual = rubric(revised)
    assert actual.pop("task_guidance") == advice
    assert actual == rubric(previous)
    assert rubric_fingerprint([], [], {}, task_guidance=advice) != rubric_fingerprint([], [], {})

    sample = frozen()
    turns = build_turn_inputs(sample["messages"], sample["policy"], 2, task_guidance=advice)
    old_turn = turn_request(sample, 0, [])
    new_turn = turn_request(turns, 0, [])
    assert {key: value for key, value in new_turn.items() if key != "task_guidance"} == old_turn
    first = build_messages(**new_turn)
    next_turn = build_messages(**turn_request(turns, 1, []))
    assert list(json.loads(first[1]["content"]))[:4] == [
        "request_version", "policy", "task_guidance", "history",
    ]
    assert first[1]["content"].count(advice) == next_turn[1]["content"].count(advice) == 1
    assert "hidden future marker" not in first[1]["content"] + next_turn[1]["content"]
    client = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    assert client.cache_identity(revised) != client.cache_identity(previous)
    assert client.cache_identity(first) != client.cache_identity(build_messages(**old_turn))
