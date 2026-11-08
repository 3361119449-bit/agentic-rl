from copy import deepcopy

import pytest
from test_grounded_policy_v3 import inputs, raw_result

from tau2_agentic_rl.schemas import JudgeResult

SCOPE = "t:policy:transfer_scope_and_message"


def transfer_case(executed=False):
    view = inputs()
    view.update(
        require_policy_attribution=True,
        transfer_validation_version="transfer-policy-v2",
    )
    view["mandatory_policy_checks"] = [{"criterion_id": SCOPE}]
    data = raw_result()
    data["mandatory_policy_checks"] = data["mandatory_policy_checks"][:1]
    check = data["mandatory_policy_checks"][0]
    check.update(criterion_id=SCOPE, violation_assistant_turn_ids=[2])
    data["policy_attributions"] = data["policy_attributions"][1:]
    data["policy_attributions"][0]["criterion_id"] = SCOPE
    data["transfer_check"] = {
        "applicable": True,
        "valid": False,
        "policy_required": not executed,
        "self_caused_only": True,
        "evidence_turn_ids": [],
        "short_reason": "Prior transfer hypothesis.",
    }
    if executed:
        view["trajectory"]["tool_events"] = [
            {"name": "transfer_to_human_agents", "success": True, "turn_id": 2}
        ]
    return view, JudgeResult.model_validate(data)


def test_unannotated_native_actor_fails_closed_instead_of_citing_future():
    from tau2_agentic_rl.judge.grounded_policy import bind_undelivered

    native = [
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "$100"},
        {"role": "user", "content": "later $999"},
    ]
    with pytest.raises(ValueError, match="annotat"):
        bind_undelivered(
            {"messages": native, "tool_events": []},
            [{"role": "assistant", "turn_idx": 1, "content": "$100"}],
        )


def test_non_procredit_live_path_still_annotates_current_actor(scratch_dir):
    from test_procredit_rollout import rollout_fixture

    _, _, record = rollout_fixture(scratch_dir, grounded_judge=True, natural_stop_at=1)
    delivered = [m for m in record.environment_transcript if m["role"] == "assistant"]
    assert delivered[0]["assistant_turn_id"] == 1
    assert not record.scoring_inputs["judge"]["trajectory"].get(
        "undelivered_generations"
    )


@pytest.mark.parametrize("executed", [False, True])
def test_scoped_rejection_never_infers_transfer_facts_and_fresh_facts_validate(
    executed,
):
    from tau2_agentic_rl.judge.client import DeepSeekJudge
    from tau2_agentic_rl.judge.grounded_policy import reconcile_witnesses
    from tau2_agentic_rl.judge.grounded_transfer import (
        TransferFactsResult,
        apply_transfer_review,
    )

    view, result = transfer_case(executed)
    DeepSeekJudge._validate_requested_criteria(result, view)
    review = {
        "criterion_id": SCOPE,
        "verdict": "rejected",
        "reason": "Scope blame was unsupported.",
    }
    revised = reconcile_witnesses(result, 2, [review], view)
    assert revised.transfer_check == result.transfer_check
    fresh = TransferFactsResult.model_validate(
        {
            "transfer_check": {
                "applicable": executed,
                "valid": executed,
                "policy_required": False,
                "self_caused_only": False,
                "evidence_turn_ids": [2] if executed else [],
                "short_reason": "Independently reviewed actual execution and current policy facts.",
            },
            "transfer_semantic_checks": [],
        }
    )
    revised = apply_transfer_review(revised, fresh, view)
    DeepSeekJudge._validate_requested_criteria(revised, view)
    assert revised.transfer_check.policy_required is False
    assert revised.transfer_check.valid is executed


def test_invalid_executed_transfer_requires_localized_scope_witness():
    from tau2_agentic_rl.judge.client import DeepSeekJudge

    view, result = transfer_case(True)
    result.mandatory_policy_checks[0].passed = True
    result.mandatory_policy_checks[0].violation_assistant_turn_ids = []
    result.policy_attributions = []
    with pytest.raises(ValueError, match="transfer"):
        DeepSeekJudge._validate_requested_criteria(result, view)


@pytest.mark.asyncio
async def test_transfer_scope_correction_requires_fresh_typed_facts_and_replays(
    monkeypatch,
):
    from test_grounded_policy_v3 import _assessment, _turn_inputs

    from tau2_agentic_rl.judge import grounded, grounded_transfer
    from tau2_agentic_rl.judge.grounded_transfer import TransferFactsResult
    from tau2_agentic_rl.reward.grounded import GroundedAssessment
    from tau2_agentic_rl.schemas import TransferCheck

    view, result = transfer_case(False)
    calls = []

    class Client:
        async def evaluate(self, **request):
            return (
                GroundedAssessment.model_validate(
                    _assessment(request, bad=request["turn_index"] == 0)
                ),
                "raw",
                "prompt",
                "cache",
            )

    class FactsClient:
        async def evaluate(self, **request):
            calls.append(request)
            facts = TransferFactsResult(
                transfer_check=TransferCheck(
                    applicable=False,
                    valid=False,
                    policy_required=False,
                    self_caused_only=False,
                    short_reason="No independently required transfer.",
                ),
                transfer_semantic_checks=[],
            )
            return facts, "facts_raw", "facts_prompt", "facts_cache"

    monkeypatch.setattr(grounded, "client_from", lambda judge: Client())
    monkeypatch.setattr(grounded_transfer, "client_from", lambda judge: FactsClient())
    revised, prepared = await grounded.prepare_grounded_turns(
        _turn_inputs(), view, result, object()
    )
    assert (
        len(calls) == 1
        and prepared["transfer_review"]["assessment"]["transfer_check"][
            "policy_required"
        ]
        is False
    )
    assert revised.transfer_check.policy_required is False
    assert grounded.validate_prepared(_turn_inputs(), view, revised, prepared)
    corrupted = deepcopy(prepared)
    corrupted.pop("transfer_review")
    with pytest.raises(ValueError):
        grounded.validate_prepared(_turn_inputs(), view, revised, corrupted)


def test_explicit_initial_history_is_context_and_generated_actors_still_required():
    from tau2_agentic_rl.judge.grounded_policy import actor_views, bind_undelivered

    native = [
        {"role": "assistant", "content": "Welcome"},
        {"role": "user", "content": "help"},
        {"role": "assistant", "assistant_turn_id": 1, "content": "Current reply"},
    ]
    bounded = [{"role": "assistant", "turn_idx": 1, "content": "Current reply"}]
    view = bind_undelivered(
        {"messages": native, "tool_events": [], "initial_prefix_length": 2}, bounded
    )
    assert [x["assistant_turn_id"] for x in actor_views(view)] == [1]
    native[2].pop("assistant_turn_id")
    with pytest.raises(ValueError):
        bind_undelivered({"messages": native, "initial_prefix_length": 2}, bounded)


def test_partial_delivery_failure_preserves_original_unscored_audit(scratch_dir):
    from test_procredit_rollout import rollout_fixture

    _, _, record = rollout_fixture(
        scratch_dir, grounded_judge=True, fail_after_delivery=True
    )
    assert record.metadata["failure_phase"] == "tau2_text_step"
    assert record.custom_reward is None and record.judge_result is None
    assert not record.metadata.get("judge_result_provenance")
