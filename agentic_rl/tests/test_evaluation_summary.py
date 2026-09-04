import json
from pathlib import Path

from scripts.summarize_evaluation import summarize


def test_infrastructure_failures_are_excluded_from_pass_at_k() -> None:
    records = Path("outputs/test_evaluation_summary/infra_exclusion")
    records.mkdir(parents=True, exist_ok=True)
    valid = {
        "task_id": "2",
        "official_scores": {"reward": 1.0},
        "custom_reward": {"strict_success": 1.0},
    }
    infrastructure_failure = {
        "task_id": "2",
        "termination_reason": "infrastructure_error",
        "official_scores": None,
        "custom_reward": None,
    }
    (records / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    (records / "infra.json").write_text(
        json.dumps(infrastructure_failure), encoding="utf-8"
    )

    result = summarize(records)
    row = result["per_task"][0]
    assert row["attempted_samples"] == 2
    assert row["samples"] == 1
    assert row["infrastructure_failures"] == 1
    assert row["official_pass1"] == 1.0
    assert row["official_pass4"] is None
