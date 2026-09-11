"""Regression tests for official split, score validation, and float boundaries."""

import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_evaluation_summary import setup_evaluation, write_sample
from test_sft_pipeline_regressions import complete_results, load_script

from scripts.summarize_evaluation import pass_hat_k, summarize
from scripts.train_airline_grpo import TAU2_COMMIT as LAUNCHER_TAU2_COMMIT
from tau2_agentic_rl.evaluation import evaluation_coverage, initialize_evaluation
from tau2_agentic_rl.pass_metrics import (
    OFFICIAL_TEST_IDS,
    TAU2_COMMIT,
    official_success,
)
from tau2_agentic_rl.versions import sha256_json


@pytest.fixture(scope="module")
def reporter():
    return load_script("report_pass1_pass4")


def test_shared_contract_matches_launcher_and_standalone(reporter):
    assert TAU2_COMMIT == LAUNCHER_TAU2_COMMIT == reporter.EXPECTED_TAU2_COMMIT
    assert reporter.OFFICIAL_TEST_IDS is OFFICIAL_TEST_IDS
    assert reporter.official_success is official_success


def test_formula_against_enumerated_successful_subsets(reporter):
    cases = 0
    for n in range(1, 13):
        for successes in range(n + 1):
            outcomes = [True] * successes + [False] * (n - successes)
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(outcomes, k))
                expected = sum(all(subset) for subset in subsets) / len(subsets)
                assert pass_hat_k(n, successes, k) == expected
                assert reporter.pass_hat_k(n, successes, k) == expected
                cases += 1
    assert cases == 728


@pytest.mark.parametrize(
    "invalid",
    [
        "wrong_twenty",
        "duplicate",
        "integer_ids",
        "missing_commit",
        "wrong_commit",
        "wrong_split",
        "wrong_n",
        "float_n",
        "bool_n",
    ],
)
def test_invalid_official_plan_rejected_on_start_and_offline_read(scratch_dir, invalid):
    root, records, manifest = setup_evaluation(scratch_dir)
    identity = manifest["identity"]
    if invalid == "wrong_twenty":
        identity["task_ids"] = [str(i) for i in range(20)]
    elif invalid == "duplicate":
        identity["task_ids"][0] = identity["task_ids"][1]
    elif invalid == "integer_ids":
        identity["task_ids"] = list(map(int, identity["task_ids"]))
    elif invalid == "missing_commit":
        identity.pop("tau2_commit")
    elif invalid == "wrong_commit":
        identity["tau2_commit"] = "unverified"
    elif invalid == "wrong_split":
        identity["record_split"] = "train"
    else:
        identity["samples_per_task"] = {
            "wrong_n": 8,
            "float_n": 4.5,
            "bool_n": True,
        }[invalid]
    # Recompute the hash: checking the digest alone must not accept this plan.
    manifest["manifest_id"] = sha256_json(identity)
    (root / "evaluation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    new_run = scratch_dir / "must-not-create"
    with pytest.raises(ValueError):
        initialize_evaluation(new_run, identity, resume=False)
    assert not new_run.exists()
    with pytest.raises(ValueError):
        evaluation_coverage(records, manifest)
    with pytest.raises(ValueError):
        summarize(records, allow_incomplete=True)


INVALID_SCORES = [
    float("nan"),
    float("inf"),
    -float("inf"),
    -0.1,
    2.0,
    True,
    False,
    None,
    "1",
    {},
    [],
    10**1000,
]


@pytest.mark.parametrize("reward", INVALID_SCORES)
@pytest.mark.parametrize(
    "field,key", [("official_scores", "reward"), ("custom_reward", "strict_success")]
)
def test_main_rejects_invalid_scores_even_for_partial_reports(
    scratch_dir,
    field,
    key,
    reward,
):
    _, records, manifest = setup_evaluation(scratch_dir, count=1)
    path = write_sample(records, manifest, 0, 0)
    row = json.loads(path.read_text(encoding="utf-8"))
    row[field][key] = reward
    path.write_text(json.dumps(row), encoding="utf-8")
    for allow_incomplete in (False, True):
        with pytest.raises(ValueError, match=key):
            summarize(records, allow_incomplete=allow_incomplete)


@pytest.mark.parametrize("reward", INVALID_SCORES)
def test_standalone_rejects_same_invalid_scores(reporter, reward):
    with pytest.raises(ValueError, match="official reward"):
        reporter.successful({"reward_info": {"reward": reward}})


@pytest.mark.parametrize("field", ["official_scores", "custom_reward"])
@pytest.mark.parametrize("payload", [[], 1, "bad", {}])
def test_score_payload_shape_is_validated(scratch_dir, field, payload):
    _, records, manifest = setup_evaluation(scratch_dir, count=1)
    path = write_sample(records, manifest, 0, 0)
    row = json.loads(path.read_text(encoding="utf-8"))
    row[field] = payload
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        evaluation_coverage(records, manifest)


def test_scoring_pending_invalid_reward_is_not_retried_as_a_new_interaction(
    scratch_dir,
):
    _, records, manifest = setup_evaluation(scratch_dir, count=1)
    path = write_sample(records, manifest, 0, 0)
    row = json.loads(path.read_text(encoding="utf-8"))
    row["official_scores"]["reward"] = float("nan")
    row["custom_reward"] = None
    row["metadata"]["failure_phase"] = "judge"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="official_scores.reward"):
        evaluation_coverage(records, manifest)


@pytest.mark.parametrize(
    "reward",
    [
        0.0,
        0.5,
        math.nextafter(1 - 1e-6, 0),
        1 - 1e-6,
        math.nextafter(1 - 1e-6, 1),
        1 - 5e-7,
        1.0,
    ],
)
def test_success_matches_pinned_official_interval_including_boundary(reporter, reward):
    expected = (1 - 1e-6) <= reward <= (1 + 1e-6)
    assert official_success(reward) == expected
    assert reporter.successful({"reward_info": {"reward": reward}}) == expected


def test_full_reports_match_at_lower_tolerance_boundary(scratch_dir, reporter):
    _, records, manifest = setup_evaluation(scratch_dir)
    source = complete_results()
    source["simulations"][0]["reward_info"]["reward"] = 1 - 1e-6
    for simulation in source["simulations"]:
        path = write_sample(
            records, manifest, simulation["task_id"], simulation["trial"]
        )
        row = json.loads(path.read_text(encoding="utf-8"))
        row["official_scores"] = simulation["reward_info"]
        row["custom_reward"]["strict_success"] = 0.0
        path.write_text(json.dumps(row), encoding="utf-8")
    metrics = summarize(records)["aggregate"]
    assert metrics == {
        "official_pass1": 1.0,
        "official_pass4": 1.0,
        "custom_strict_pass1": 0.0,
        "custom_strict_pass4": 0.0,
    }
    assert reporter.compute(source["simulations"]) == {"pass^1": 1.0, "pass^4": 1.0}


@pytest.mark.parametrize("invalid", [False, True])
def test_standalone_cli_needs_no_installed_rl_or_gpu_dependencies(scratch_dir, invalid):
    source = complete_results()
    source["simulations"][0]["reward_info"]["reward"] = (
        float("nan") if invalid else 1 - 1e-6
    )
    data, output = scratch_dir / "results.json", scratch_dir / "metrics.json"
    data.write_text(json.dumps(source), encoding="utf-8")
    script = (
        Path(__file__).parents[2] / "training/tau2_rollout_sft/report_pass1_pass4.py"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            str(script),
            str(data),
            "--output-json",
            str(output),
        ],
        cwd=scratch_dir,
        env=environment,
        capture_output=True,
        text=True,
    )
    if invalid:
        assert result.returncode != 0
        assert "official reward" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text(encoding="utf-8")) == {
            "pass^1": 1,
            "pass^4": 1,
        }
