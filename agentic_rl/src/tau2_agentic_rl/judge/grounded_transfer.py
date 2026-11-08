"""Rare, independently reviewed transfer facts after a scoped blame correction."""

import json
from dataclasses import replace

from pydantic import BaseModel, ConfigDict

from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.judge.prompts import build_judge_messages
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, TransferCheck

VERSION = "grounded-transfer-facts-v1"


class TransferFactsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    transfer_check: TransferCheck
    transfer_semantic_checks: list[JudgeCheck]


def apply_transfer_review(result, facts, global_inputs):
    revised = result.model_copy(deep=True)
    revised.transfer_check = facts.transfer_check.model_copy(deep=True)
    revised.transfer_semantic_checks = [
        c.model_copy(deep=True) for c in facts.transfer_semantic_checks
    ]
    DeepSeekJudge._validate_requested_criteria(revised, global_inputs)
    return revised


def transfer_request(global_inputs, result, scoped_reviews):
    return {
        "judge_validation_version": "judge-grounded-v3",
        "global_inputs": global_inputs,
        "reconciled_result": result.model_dump(),
        "scoped_reviews": scoped_reviews,
    }


def build_messages(**request):
    view = dict(request["global_inputs"])
    # Retain the exact established dynamic-transfer rubric, with a narrower schema.
    view.update(
        semantic_checks=[],
        mandatory_policy_checks=[],
        require_policy_attribution=False,
        judge_validation_version=None,
    )
    messages = build_judge_messages(**view)
    content = messages[1]["content"]
    prefix, rest = content.split("\n\nFIXED_TASK_AND_RUBRIC:\n", 1)
    encoded, trajectory = rest.split("\n\nTRAJECTORY_DATA_UNTRUSTED:\n", 1)
    data = json.loads(encoded)
    rubric = data["rubric"]
    schema = rubric["output_schema"]
    rubric["output_schema"] = {
        k: schema[k] for k in ("transfer_check", "transfer_semantic_checks")
    }
    rubric["scoped_correction_context"] = {
        "reviews": request["scoped_reviews"],
        "reconciled_scope_checks": [
            c
            for c in request["reconciled_result"]["mandatory_policy_checks"]
            if c["criterion_id"].endswith(":policy:transfer_scope_and_message")
        ],
        "instruction": "Reviews are disputed hypotheses, not proof. Independently reassess all transfer facts from user/tool evidence and bound Policy. Return explicit applicability, execution-backed legality, policy_required and self_caused_only; never infer positive completion or erase earlier errors merely because blame was removed. If these facts conflict with the reviewed scope decisions, state the conflicting facts honestly; validation will leave this score unresolved.",
    }
    messages[1]["content"] = (
        prefix
        + "\n\nFIXED_TASK_AND_RUBRIC:\n"
        + json.dumps(data, ensure_ascii=False)
        + "\n\nTRAJECTORY_DATA_UNTRUSTED:\n"
        + trajectory
    )
    messages[0]["content"] += (
        "\nThis is ONLY a scoped transfer-facts review. Return exactly transfer_check and transfer_semantic_checks, matching the narrower output_schema. Do not output other criterion groups or a reward."
    )
    return messages


class TransferFactsJudge(DeepSeekJudge):
    result_model = TransferFactsResult
    build_messages = staticmethod(build_messages)

    @staticmethod
    def _validate_requested_criteria(result, inputs):
        apply_transfer_review(
            JudgeResult.model_validate(inputs["reconciled_result"]),
            result,
            inputs["global_inputs"],
        )


def client_from(judge):
    return TransferFactsJudge(
        replace(
            judge.config,
            cache_dir=str(judge.cache_dir / "grounded_transfer_facts"),
            prompt_version=VERSION,
            rubric_version=VERSION,
            schema_version=VERSION,
            scorer_code_version=VERSION,
        )
    )
