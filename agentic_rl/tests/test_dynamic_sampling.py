from tau2_agentic_rl.dynamic_sampling import (
    ScoredRollout,
    grpo_advantages,
    select_valid_groups,
)


def rollout(task: str, index: int, reward: float) -> ScoredRollout:
    return ScoredRollout(task, f"{task}-{index}", reward)


def test_only_complete_mixed_groups_are_selected() -> None:
    rows = [rollout("a", i, i % 2) for i in range(8)]
    rows += [rollout("b", i, 1.0) for i in range(8)]
    rows += [rollout("c", i, i % 2) for i in range(7)]
    selected = select_valid_groups(rows, group_size=8, target_groups=1)
    assert len(selected) == 1
    assert {item.task_id for item in selected[0]} == {"a"}


def test_insufficient_valid_groups_returns_no_optimizer_batch() -> None:
    rows = [rollout("a", i, i % 2) for i in range(8)]
    assert select_valid_groups(rows, group_size=8, target_groups=4) == []


def test_grpo_advantages_are_centered() -> None:
    values = grpo_advantages([rollout("a", 0, 0.0), rollout("a", 1, 1.0)])
    assert abs(sum(values.values())) < 1e-12
