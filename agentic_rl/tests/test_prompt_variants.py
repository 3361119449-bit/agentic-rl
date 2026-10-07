import json
from pathlib import Path

from tau2_agentic_rl.agent_policy import (
    extract_airline_policy,
    load_agent_system_prompt,
    prompt_sha256,
)
from tau2_agentic_rl.config import load_yaml


ROOT = Path(__file__).parents[1]
BASE_TRAIN_CONFIG = ROOT / "configs/rl/airline_grpo_v1.yaml"
GROUNDED_TRAIN_CONFIG = (
    ROOT / "configs/rl/airline_grpo_grounding_examples_v1.yaml"
)
GROUNDED_EVAL_CONFIG = (
    ROOT / "configs/evaluation/airline_eval_grounding_examples_v1.yaml"
)

EXPECTED_EXAMPLE_FAMILIES = [
    "tool_error_is_not_success",
    "authoritative_tool_state",
    "no_unsupported_facts",
    "correct_before_retry",
    "schema_tools_only",
    "payment_validation",
    "confirmation_before_write",
    "tool_call_dependencies",
]


def test_grounding_prompt_variant_is_selectable_without_changing_policy():
    base_config = load_yaml(BASE_TRAIN_CONFIG)
    grounded_train = load_yaml(GROUNDED_TRAIN_CONFIG)
    grounded_eval = load_yaml(GROUNDED_EVAL_CONFIG)

    base_prompt = load_agent_system_prompt(base_config, ROOT)
    grounded_prompt = load_agent_system_prompt(grounded_train, ROOT)

    assert load_agent_system_prompt(grounded_eval, ROOT) == grounded_prompt
    assert grounded_prompt != base_prompt
    assert extract_airline_policy(grounded_prompt) == extract_airline_policy(base_prompt)

    artifact_path = ROOT / grounded_train["agent_policy"]["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["variant"] == "grounding_examples_v1"
    assert artifact["base_system_prompt_sha256"] == prompt_sha256(base_prompt)
    assert artifact["example_families"] == EXPECTED_EXAMPLE_FAMILIES
    assert artifact["system_prompt_sha256"] == prompt_sha256(grounded_prompt)
    assert grounded_train["agent_policy"] == grounded_eval["agent_policy"]
