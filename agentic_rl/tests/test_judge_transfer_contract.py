import asyncio
import json

import pytest
from test_judge_evidence import install_mock_api

from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize(
    "applicable,valid", [(False, True), (True, True), (True, False)]
)
def test_unexecuted_transfer_is_normalized_without_losing_rollout(
    scratch_dir, monkeypatch, required, applicable, valid
):
    payload = {
        "transfer_check": {
            "applicable": applicable,
            "valid": valid,
            "evidence_turn_ids": [987],
            "short_reason": "model guessed a transfer",
        }
    }
    calls = install_mock_api(monkeypatch, [payload])
    judge = DeepSeekJudge(
        JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1)
    )
    inputs = dict(
        task={},
        policy="fixed",
        semantic_checks=[],
        mandatory_policy_checks=[],
        trajectory={
            "messages": [],
            "tool_events": [
                {"name": "transfer_to_human_agents", "success": False, "turn_id": 1}
            ],
        },
        transfer_rule={"allowed": required, "required": required},
    )
    result, raw, *identity = asyncio.run(judge.evaluate(**inputs))
    assert len(calls) == 1
    assert result.transfer_check.applicable is required
    assert result.transfer_check.valid is False
    assert result.transfer_check.evidence_turn_ids == []
    assert json.loads(raw)["transfer_check"] == payload["transfer_check"]
    assert asyncio.run(judge.evaluate(**inputs))[0] == result
    assert len(calls) == 1


def test_executed_transfer_cannot_be_promoted_from_not_applicable(
    scratch_dir, monkeypatch
):
    calls = install_mock_api(
        monkeypatch,
        [
            {
                "transfer_check": {
                    "applicable": False,
                    "valid": True,
                    "evidence_turn_ids": [1],
                }
            }
        ],
    )
    judge = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    result, *_ = asyncio.run(
        judge.evaluate(
            task={},
            policy="fixed",
            semantic_checks=[],
            mandatory_policy_checks=[],
            trajectory={
                "messages": [],
                "tool_events": [
                    {"name": "transfer_to_human_agents", "success": True, "turn_id": 1}
                ],
            },
            transfer_rule={"allowed": True},
        )
    )
    assert len(calls) == 1
    assert result.transfer_check.applicable is True
    assert result.transfer_check.valid is False
