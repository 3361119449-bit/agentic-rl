import sys
from pathlib import Path

import pytest

from scripts import profile_sft_baseline
from scripts.profile_rollouts import profile_records


def _record(task_id: str, reward: float) -> dict:
    return {
        "task_id": task_id,
        "assistant_turns": 2,
        "trajectory_tokens": 100,
        "termination_reason": "agent_stop",
        "tool_events": [],
        "official_scores": {"reward": reward},
        "custom_reward": {
            "train_reward": reward,
            "strict_success": reward,
            "branch": "normal",
            "components": {},
        },
    }


def test_profile_detects_mixed_and_all_zero_groups() -> None:
    rows = [_record("1", float(index % 2)) for index in range(8)]
    rows += [_record("2", 0.0) for _ in range(8)]
    result = profile_records(rows)
    assert result["complete_group_count"] == 2
    assert result["mixed_group_rate"] == 0.5
    assert result["all_zero_group_rate"] == 0.5


@pytest.mark.parametrize("tag", ["areal_then_tau2", "run with spaces", "../trial"])
def test_profile_wrapper_reuses_effective_evaluation_options(monkeypatch, tag):
    calls = []
    monkeypatch.setattr(
        profile_sft_baseline.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr(sys, "argv", ["profile", "--tag", tag, "--samples", "4"])
    profile_sft_baseline.main()
    evaluate = profile_sft_baseline.parse_args(calls[0][2:])
    profile = calls[1]
    assert Path(profile[2]).parts[-2:] == (evaluate.tag, "trajectories")
    assert Path(profile[-1]).name == f"{evaluate.tag}.json"
    assert profile[profile.index("--group-size") + 1] == str(evaluate.samples) == "4"


def test_profile_dry_run_never_reads_old_results(monkeypatch):
    calls = []
    monkeypatch.setattr(
        profile_sft_baseline.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr(sys, "argv", ["profile", "--dry-run"])
    profile_sft_baseline.main()
    assert len(calls) == 1 and "--dry-run" in calls[0]
