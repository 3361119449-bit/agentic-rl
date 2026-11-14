import json
from pathlib import Path

from scripts.train_airline_grpo import FULL_TRAIN_CONFIG, build_command
from tau2_agentic_rl.annotations import load_task_judge_guidance, load_task_mapping
from tau2_agentic_rl.config import load_yaml, validate_procredit_config
from tau2_agentic_rl.reward.required_actions import load_required_actions
from tau2_agentic_rl.training_selection import select_training_rows

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_full_train_starts_with_repaired_thirty_and_the_global_judge():
    project = load_yaml(ROOT / FULL_TRAIN_CONFIG)
    validate_procredit_config(project)
    selection = project["training_selection"]
    old = read("data/splits/airline_internal_dev.v1.json")
    split = read(selection["split_file"])
    ids = set(split["rl_train"])
    assert len(ids) == 30 and ids == set(old["rl_train"] + old["internal_dev"])
    assert not split["internal_dev"]
    assert split["official_test"] == old["official_test"]
    assert project["judge"]["enabled"] is True
    assert project["judge"]["calls_per_trajectory"] == 1

    definitions = load_task_mapping(ROOT / selection["task_repairs"])
    annotations = project["annotations"]
    required = load_required_actions(ROOT / annotations["required_actions"])
    semantic = load_task_mapping(ROOT / annotations["semantic_checks"])
    guidance = load_task_judge_guidance(ROOT, annotations, ids)
    assert required.keys() == semantic.keys() == guidance.keys() == ids
    assert {action["arguments"]["reservation_id"] for action in required["39"]} == {
        "8C8K4E", "LU15PA",
    }
    assert definitions["7"]["evaluation_criteria"]["communicate_info"] == ["708"]
    assert semantic["7"]["semantic_checks"][-1]["description"] == (
        definitions["7"]["evaluation_criteria"]["nl_assertions"][3]
    )
    assert semantic["39"]["covered_checks"][2]["covered_by"] == "db"

    _, manifest = select_training_rows(
        rows=[{"extra_info": {"task_id": tid, "split": "train"}} for tid in split["rl_train"]],
        tasks={tid: {"id": tid, **definitions.get(tid, {})} for tid in ids},
        required=required, dependencies={}, transfers={tid: {} for tid in ids},
        split=split, mode=selection["mode"], progress_version="progress-v3",
        repaired_task_ids=definitions.keys(),
        evaluator=lambda task, _: {
            "db": True,
            "communicate": [False] * len(
                task.get("evaluation_criteria", {}).get("communicate_info") or []
            ),
        },
    )
    assert set(manifest["included_task_ids"]) == ids
    assert manifest["excluded_task_ids"] == []

    command = build_command(
        project_root=ROOT, model_path="fixture", train_file=Path("train.parquet"),
        val_file=Path("dev.parquet"), total_epochs=1, extra=[], project_config=project,
    )
    assert "trainer.val_before_train=false" in command
    assert "trainer.test_freq=-1" in command
