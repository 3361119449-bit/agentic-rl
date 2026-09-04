import json
from pathlib import Path

import yaml

from tau2_agentic_rl.config import expand_env


def test_env_placeholder_default(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VALUE", raising=False)
    assert expand_env("${MISSING_VALUE:-fallback}") == "fallback"


def test_env_placeholder_override(monkeypatch) -> None:
    monkeypatch.setenv("SET_VALUE", "exact")
    assert expand_env("${SET_VALUE:-fallback}") == "exact"


def test_qwen_instruct_uses_hermes_json_tool_parser() -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/rl/airline_grpo_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["rollout"]["tool_parser"] == "hermes"


def test_every_train_task_has_a_task_scoped_policy_judge_check() -> None:
    rows = json.loads(
        (
            Path(__file__).parents[1]
            / "data/annotations/airline_mandatory_policy_rules.train.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert rows
    assert all(
        row["judge_checks"]
        and row["judge_checks"][0]["criterion_id"].startswith(f"{row['task_id']}:")
        for row in rows
    )
