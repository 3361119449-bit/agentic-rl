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
