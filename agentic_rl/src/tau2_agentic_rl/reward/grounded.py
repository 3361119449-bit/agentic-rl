"""Typed current-generation findings; no reward arithmetic is changed here."""

import ast
import math
import re
from decimal import Decimal, DecimalException
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt

from tau2_agentic_rl.reward.simple_turn import (
    GROUNDED_VERSION,
    JUDGE_VERSION,
    RefusalRequest,
    RefusalRevocation,
    validate_decision,
    validate_target_request,
)

CATEGORIES = (
    "amount",
    "authorization",
    "parameters",
    "grounding",
    "policy",
    "transfer_summary",
)
Category = Literal[
    "amount", "authorization", "parameters", "grounding", "policy", "transfer_summary"
]
Number = StrictInt | StrictFloat


class StrictAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TurnFinding(StrictAudit):
    category: Category
    verdict: Literal["ok", "error", "uncertain"]
    quote: str
    evidence_message_indices: list[StrictInt]
    reason: str


class AmountCheck(StrictAudit):
    kind: Literal["arithmetic", "authorization", "ranking"]
    expression: str | None = None
    candidates: list[Number] | None = None
    rank: StrictInt | None = None
    expected: Number
    reported: Number | None = None
    authorized: Number | None = None
    quote: str
    evidence_message_indices: list[StrictInt]
    reason: str


class PolicyReview(StrictAudit):
    criterion_id: str
    verdict: Literal["confirmed", "rejected", "uncertain"]
    quote: str
    policy_quote: str
    evidence_message_indices: list[StrictInt]
    reason: str


class GroundedAssessment(StrictAudit):
    turn_index: StrictInt
    target_message_index: StrictInt
    target_generation_sha256: str
    target_quote: str
    audited_categories: list[Category]
    checks: list[TurnFinding]
    amount_checks: list[AmountCheck]
    refusal_verified: StrictBool
    refusal_requests: list[RefusalRequest]
    revoked_refusals: list[RefusalRevocation]
    policy_reviews: list[PolicyReview]
    neutral_reason: str


def version_for_project(project):
    version = project.get("judge", {}).get("grounding_version")
    if version not in (None, "judge-grounded-v3"):
        raise ValueError("unknown Judge grounding version")
    return GROUNDED_VERSION if version else JUDGE_VERSION


def arithmetic(expression):
    if not isinstance(expression, str) or not 0 < len(expression) <= 256:
        raise ValueError("invalid arithmetic expression")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("arithmetic expression too large")

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = Decimal(str(node.value))
                if not value.is_finite():
                    raise ValueError("nonfinite arithmetic")
                return value
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                x = visit(node.operand)
                return -x if isinstance(node.op, ast.USub) else x
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                return a / b
            raise ValueError("unsupported arithmetic")

        value = visit(tree.body)
        if not value.is_finite():
            raise ValueError("nonfinite arithmetic")
        return value
    except (SyntaxError, DecimalException, OverflowError) as exc:
        raise ValueError("invalid arithmetic") from exc


def _reason_contradiction(reason):
    # This is a retry trigger, NEVER a keyword-to-bad conversion. Ignore quoted
    # text and explicit negation; only narrow affirmative verdict clauses count.
    unquoted = re.sub(r'"[^"\n]*"|`[^`\n]*`', "", reason)
    pattern = r"\b(?:this is a bad turn|i(?:'ll| will) mark bad_turn true|making it a bad turn)\b"
    for match in re.finditer(pattern, unquoted, re.I):
        before = unquoted[max(0, match.start() - 40) : match.start()].lower()
        if not re.search(
            r"\b(?:not|never|deny|denies|incorrect|quote|quoted)\b", before
        ):
            return True
    return False


def _evidence(audit, request):
    target = request["target_message_index"]
    ids = audit.evidence_message_indices
    if (
        not ids
        or len(ids) != len(set(ids))
        or target not in ids
        or any(i < 0 or i > target for i in ids)
    ):
        raise ValueError("finding must cite current target within causal prefix")
    if not audit.quote.strip() or audit.quote not in request["target_generation_text"]:
        raise ValueError("finding quote is not in current generation")
    if not audit.reason.strip():
        raise ValueError("finding requires evidence reason")


def validate_assessment(value, request):
    validate_target_request(request)
    if request["request_version"] != GROUNDED_VERSION:
        raise ValueError("grounded assessment requires v3 target identity")
    model = GroundedAssessment.model_validate(value)
    if len(model.audited_categories) != len(CATEGORIES) or set(
        model.audited_categories
    ) != set(CATEGORIES):
        raise ValueError("missing current-generation audit categories")
    errors, reasons, evidence = [], [], {request["target_message_index"]}
    for finding in model.checks:
        _evidence(finding, request)
        evidence.update(finding.evidence_message_indices)
        if finding.verdict == "error":
            errors.append(finding.category)
        elif _reason_contradiction(finding.reason):
            raise ValueError(
                "finding reason/verdict contradiction; reassess current target"
            )
        reasons.append(finding.reason)
    for item in model.amount_checks:
        _evidence(item, request)
        evidence.update(item.evidence_message_indices)
        numbers = [
            item.expected,
            item.reported,
            item.authorized,
            *(item.candidates or []),
        ]
        if any(not math.isfinite(x) for x in numbers if x is not None):
            raise ValueError("nonfinite monetary operand")
        if item.kind == "ranking":
            if (
                not item.candidates
                or type(item.rank) is not int
                or not 1 <= item.rank <= len(item.candidates)
            ):
                raise ValueError("invalid ranking operands")
            computed = Decimal(str(sorted(item.candidates)[item.rank - 1]))
            if abs(computed - Decimal(str(item.expected))) > Decimal("0.000001"):
                raise ValueError("ranking expected value is inconsistent")
        else:
            computed = arithmetic(item.expression)
            if abs(computed - Decimal(str(item.expected))) > Decimal("0.000001"):
                raise ValueError("arithmetic expected value is inconsistent")
        if item.reported is not None and abs(
            computed - Decimal(str(item.reported))
        ) > Decimal("0.005"):
            errors.append("amount")
        if item.kind == "authorization":
            if item.authorized is None or item.authorized < 0:
                raise ValueError("authorization audit needs a nonnegative user limit")
            if computed > Decimal(str(item.authorized)) + Decimal("0.005"):
                errors.append("authorization")
        reasons.append(item.reason)
    expected_reviews = {x["criterion_id"] for x in request.get("policy_candidates", [])}
    if (
        len(model.policy_reviews) != len(expected_reviews)
        or {x.criterion_id for x in model.policy_reviews} != expected_reviews
    ):
        raise ValueError("policy review coverage differs from requested candidates")
    for review in model.policy_reviews:
        _evidence(review, request)
        if (
            not review.policy_quote.strip()
            or review.policy_quote not in request["policy"]
        ):
            raise ValueError("policy review lacks exact bound policy quote")
        if review.verdict == "uncertain":
            raise ValueError("unresolved global policy review")
        if review.verdict == "confirmed":
            errors.append("policy")
        elif _reason_contradiction(review.reason):
            raise ValueError("policy review reason/verdict contradiction")
        reasons.append(review.reason)
        evidence.update(review.evidence_message_indices)
    bad = bool(errors)
    if not bad and _reason_contradiction(model.neutral_reason):
        raise ValueError(
            "neutral reason/verdict contradiction; reassess current target"
        )
    if not model.neutral_reason.strip():
        raise ValueError("neutral reason is missing")
    if model.refusal_verified and any(x.verdict == "uncertain" for x in model.checks):
        raise ValueError("uncertain facts cannot qualify a clean refusal")
    if model.refusal_verified != bool(model.refusal_requests):
        raise ValueError("refusal status and requests disagree")
    clean_refusal = model.refusal_verified and not bad
    decision = {
        "turn_index": model.turn_index,
        "target_message_index": model.target_message_index,
        "target_generation_sha256": model.target_generation_sha256,
        "target_quote": model.target_quote,
        "bad_turn": bad,
        "correct_refusal": clean_refusal,
        "reason": " ".join(reasons) if reasons else model.neutral_reason,
        "evidence_message_indices": sorted(evidence),
        "refusal_requests": [x.model_dump() for x in model.refusal_requests]
        if clean_refusal
        else [],
        "revoked_refusals": [x.model_dump() for x in model.revoked_refusals],
    }
    return model.model_dump(), validate_decision(decision, request)


def global_options(project):
    return (
        {"judge_validation_version": "judge-grounded-v3"}
        if version_for_project(project) == GROUNDED_VERSION
        else {}
    )


def frozen_options(project):
    return (
        {"judge": {"grounding_version": "judge-grounded-v3"}}
        if version_for_project(project) == GROUNDED_VERSION
        else {}
    )
