"""Keep current RL/evaluation on the actual SFT v4 one-tool contract."""

from pathlib import Path

import pytest

from tau2_agentic_rl.agent_policy import load_agent_system_prompt
from tau2_agentic_rl.config import load_yaml, validate_procredit_config

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("profile", [
    "configs/rl/airline_procredit_v4_train30.yaml",
    "configs/evaluation/airline_official80.yaml",
])
def test_current_profile_explicitly_requires_one_tool(profile):
    config = load_yaml(ROOT / profile)
    assert config["rollout"].get("max_tool_calls_per_turn") == 1


@pytest.mark.parametrize("value", [0, 2, True, "1"])
def test_current_config_rejects_other_tool_counts(value):
    config = load_yaml(ROOT / "configs/rl/airline_procredit_v4_train30.yaml")
    config["rollout"]["max_tool_calls_per_turn"] = value
    with pytest.raises(ValueError, match="max_tool_calls_per_turn"):
        validate_procredit_config(config)


def test_current_run_rejects_legacy_multitool_prompt_before_api():
    config = load_yaml(ROOT / "configs/rl/airline_procredit_v4_train30.yaml")
    config["agent_policy"] = {
        "path": "configs/prompts/areal_airline_sft.v1.json",
        "system_prompt_sha256": "e490e0859e1357ed6dd777fdbddab15b11d6de96ff7161328a6babcab77348ab",
    }
    with pytest.raises(ValueError, match="single-tool"):
        load_agent_system_prompt(config, ROOT)
