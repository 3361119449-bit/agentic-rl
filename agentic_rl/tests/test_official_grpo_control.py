from pathlib import Path

import pytest

from tau2_agentic_rl.advantages import is_procredit
from tau2_agentic_rl.config import load_yaml, official_grpo_profile
from tau2_agentic_rl.procredit_runtime import build_official_grpo_tensors
from tau2_agentic_rl.training_config import effective_project_config, training_overrides


def test_official_profile_matches_formal_rollout_training_and_task_selection():
    root = Path(__file__).parents[1]
    formal = load_yaml(root / "configs/rl/airline_procredit_v4_train30.yaml")
    official = effective_project_config(official_grpo_profile(formal), [])
    for field in ("algorithm", "rollout", "ppo", "optimizer", "precision",
                  "training_selection", "dynamic_sampling", "slot_recovery",
                  "user_sim_filter", "user_simulator", "agent_policy"):
        assert official[field] == formal[field]
    assert not is_procredit(official)
    assert official["judge"] == {"enabled": False}
    overrides = training_overrides(formal)
    overrides.pop("+algorithm.procredit_enabled")
    overrides.pop("algorithm.gamma")
    assert training_overrides(official) == overrides


@pytest.mark.parametrize("nested", [False, True])
def test_official_score_is_the_only_advantage_on_actor_tokens(nested):
    torch = pytest.importorskip("torch")
    keys = ["group_0_0", "group_5_0", "pad_0_0"]
    rows = [
        {"tau2_official_reward": score, "reward_extra_info": {"train_reward": score},
         "task_id": "9", "policy_version": 1, "trajectory_id": key}
        for key, score in zip(keys[:2], [1.0, 0.0], strict=True)
    ]
    mask_rows = [torch.tensor(row) for row in ([1, 0, 1], [1, 1, 0], [0, 0, 0])]
    masks = (torch.nested.as_nested_tensor(mask_rows, layout=torch.jagged)
             if nested else torch.stack(mask_rows))
    result = build_official_grpo_tensors(
        keys, [*rows, {}], masks, padding=[False, False, True],
        terminal_groups={"group": {"status": "failure", "rollout_n": 8,
                                   "failed_session_ids": [1, 2, 3, 4, 6, 7],
                                   "failure_kind": "transient_exhausted"}},
    )
    adv = 0.5 / (0.5 + 1e-6)
    expected = [[adv, 0, adv], [-adv, -adv, 0], [0, 0, 0]]
    for values, truth in zip(result["advantages"].unbind(), expected, strict=True):
        assert values.tolist() == pytest.approx(truth)
