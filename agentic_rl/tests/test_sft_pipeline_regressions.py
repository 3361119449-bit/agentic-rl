"""Standalone conversion/reporting contracts, included in both tokenizer CI jobs."""

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEST_IDS = {
    "2",
    "6",
    "8",
    "13",
    "16",
    "18",
    "19",
    "22",
    "24",
    "25",
    "26",
    "29",
    "30",
    "31",
    "32",
    "35",
    "37",
    "44",
    "45",
    "48",
}
COMMIT = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"


def load_script(name):
    path = ROOT / "training/tau2_rollout_sft" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converter():
    return load_script("convert_tau2_results_to_sft")


@pytest.fixture(scope="module")
def reporter():
    return load_script("report_pass1_pass4")


@pytest.fixture(scope="module")
def tokenizer():
    source = os.environ.get("QWEN_TOKENIZER_PATH")
    if not source:
        pytest.skip("set QWEN_TOKENIZER_PATH to the pinned local Qwen tokenizer")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, local_files_only=True)


def complete_results(n=4):
    return {
        "info": {
            "git_commit": COMMIT,
            "num_trials": n,
            "environment_info": {"domain_name": "airline"},
        },
        "tasks": [{"id": task} for task in sorted(TEST_IDS, key=int)],
        "simulations": [
            {
                "id": f"{task}-{trial}",
                "task_id": task,
                "trial": trial,
                "termination_reason": "user_stop",
                "reward_info": {"reward": 1.0},
            }
            for task in sorted(TEST_IDS, key=int)
            for trial in range(n)
        ],
    }


def test_pinned_test_ids_match_repository_split(reporter):
    split = json.loads(
        (ROOT / "agentic_rl/data/splits/airline_internal_dev.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(reporter.OFFICIAL_TEST_IDS) == TEST_IDS == set(split["official_test"])


def test_token_count_is_full_chat_length_not_encoding_field_count(converter, tokenizer):
    messages = [
        {"role": "system", "content": "Follow the airline policy. " * 30},
        {"role": "user", "content": "Please look up my booking."},
        {"role": "assistant", "content": "airline " * 17_000},
    ]
    expected = len(
        tokenizer.apply_chat_template(
            messages,
            tools=[],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    assert expected > 16_384
    assert converter.token_length(tokenizer, messages, []) == expected


def test_conversion_keeps_short_prefix_but_filters_long_answer(
    converter,
    tokenizer,
    scratch_dir,
    monkeypatch,
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_reservation_details",
                "description": "Look up a reservation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {"type": "string"},
                    },
                },
            },
        }
    ]
    source = [
        {"role": "user", "content": "Find ABC123."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": "ABC123"},
                }
            ],
        },
        {"role": "tool", "content": '{"status":"available"}'},
        {"role": "assistant", "content": "airline " * 17_000},
    ]
    prompt = "Follow the airline policy. " * 30
    results = {
        "info": {"environment_info": {"domain_name": "airline"}},
        "simulations": [
            {
                "id": "train-0",
                "task_id": "0",
                "trial": 0,
                "reward_info": {"reward": 1},
                "termination_reason": "user_stop",
                "messages": source,
            }
        ],
    }
    results_path, context_path, output = [
        scratch_dir / name for name in ("results.json", "context.json", "sft.jsonl")
    ]
    results_path.write_text(json.dumps(results), encoding="utf-8")
    context_path.write_text(
        json.dumps(
            {
                "domain": "airline",
                "system_prompt": prompt,
                "tools": tools,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(converter, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "--results",
            str(results_path),
            "--context",
            str(context_path),
            "--output",
            str(output),
        ],
    )
    converter.main()
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    ids = tokenizer.apply_chat_template(
        row["messages"] + [row["answer"]],
        tools=tools,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assert 2 < row["metadata"]["token_count"] == len(ids) <= 16_384
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counters"]["rejected_over_16k_rows"] == 1


def test_standalone_report_rejects_missing_whole_task(reporter):
    rows = [r for r in complete_results()["simulations"] if r["task_id"] != "2"]
    with pytest.raises(ValueError, match="incomplete"):
        reporter.compute(rows)


@pytest.mark.parametrize("kind", ["trial", "id", "foreign_task", "invalid_trial"])
def test_standalone_report_rejects_ambiguous_samples(reporter, kind):
    rows = complete_results()["simulations"]
    if kind == "trial":
        rows[1]["trial"] = rows[0]["trial"]
    elif kind == "id":
        rows[1]["id"] = rows[0]["id"]
    elif kind == "foreign_task":
        rows[0]["task_id"] = "0"
    else:
        rows[0]["trial"] = True
    with pytest.raises(ValueError):
        reporter.compute(rows)


def test_standalone_report_requires_all_declared_trials(reporter):
    rows = complete_results(n=8)["simulations"]
    assert reporter.compute(rows, num_trials=8) == {"pass^1": 1, "pass^4": 1}
    with pytest.raises(ValueError, match="incomplete"):
        reporter.compute(rows[:-1], num_trials=8)


def test_infrastructure_retry_fills_slot_without_counting_twice(reporter):
    rows = complete_results()["simulations"]
    replacement = deepcopy(rows[0])
    replacement["id"] = "replacement"
    rows[0].update(termination_reason="infrastructure_error", reward_info=None)
    with pytest.raises(ValueError, match="incomplete"):
        reporter.compute(rows)
    assert reporter.compute(rows + [replacement]) == {"pass^1": 1, "pass^4": 1}


@pytest.mark.parametrize("reward", [None, float("nan"), True])
def test_unscored_or_invalid_reward_is_not_silently_counted(reporter, reward):
    rows = complete_results()["simulations"]
    rows[0]["reward_info"] = {"reward": reward}
    with pytest.raises(ValueError, match="reward"):
        reporter.compute(rows)


@pytest.mark.parametrize("directory_format", [False, True])
def test_report_cli_reads_both_official_formats(
    reporter,
    scratch_dir,
    monkeypatch,
    directory_format,
):
    root = complete_results()
    # Half the tasks: 4/4; the other half: 2/4. Macro metrics are 0.75 and 0.5.
    for row in root["simulations"]:
        if int(row["task_id"]) >= 26:
            row["reward_info"]["reward"] = float(row["trial"] < 2)
    if directory_format:
        folder = scratch_dir / "simulations"
        folder.mkdir()
        for row in root["simulations"]:
            (folder / f"{row['id']}.json").write_text(json.dumps(row), encoding="utf-8")
        root["simulations"] = []
    source, output = scratch_dir / "results.json", scratch_dir / "metrics.json"
    source.write_text(json.dumps(root), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            str(scratch_dir if directory_format else source),
            "--output-json",
            str(output),
        ],
    )
    reporter.main()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "pass^1": 0.75,
        "pass^4": 0.5,
    }


@pytest.mark.parametrize("invalid", ["domain", "commit", "tasks", "num_trials"])
def test_report_cli_rejects_foreign_or_partial_metadata_before_writing(
    reporter,
    scratch_dir,
    monkeypatch,
    invalid,
):
    root = complete_results()
    if invalid == "domain":
        root["info"]["environment_info"]["domain_name"] = "retail"
    elif invalid == "commit":
        root["info"]["git_commit"] = "unverified"
    elif invalid == "tasks":
        root["tasks"] = root["tasks"][:-1]
    else:
        root["info"]["num_trials"] = 8
    source, output = scratch_dir / "results.json", scratch_dir / "metrics.json"
    source.write_text(json.dumps(root), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["report", str(source), "--output-json", str(output)]
    )
    with pytest.raises(ValueError):
        reporter.main()
    assert not output.exists()
