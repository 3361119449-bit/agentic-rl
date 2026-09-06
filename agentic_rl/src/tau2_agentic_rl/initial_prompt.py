"""Untruncated runtime prompts, not the placeholder stored in RL Parquet."""

from __future__ import annotations

from collections import Counter
from math import ceil

from tau2_agentic_rl.budget import ContextBudget
from tau2_agentic_rl.tooling import CONFIRMATION_PROTOCOL


def initial_messages(policy: str, incoming: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": f"{policy}\n\n{CONFIRMATION_PROTOCOL}"},
        *incoming,
    ]


def encode_full_chat(tokenizer, messages, *, tools=None, template_kwargs=None):
    """The text-only Qwen path; never invoke veRL's left-truncating helper."""
    kwargs = dict(template_kwargs or {})
    reserved = {
        "tokenize",
        "tools",
        "add_generation_prompt",
        "truncation",
        "max_length",
        "padding",
        "return_dict",
        "return_tensors",
        "tokenizer_kwargs",
    }
    if reserved.intersection(kwargs):
        raise ValueError(
            "chat template kwargs cannot override untruncated tokenization"
        )
    return list(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
            **kwargs,
        )
    )


def inspect_initial_prompt(task_id, token_count, prompt_limit, budget: ContextBudget):
    if type(prompt_limit) is not int or prompt_limit < 1:
        raise ValueError("initial prompt limit must be a positive integer")
    reserve = (
        budget.reserved_observation_tokens
        + budget.reserved_template_tokens
        + budget.min_final_response_tokens
    )
    violations = []
    if token_count > prompt_limit:
        violations.append("initial_prompt_limit_exceeded")
    if token_count + reserve > budget.max_context_tokens:
        violations.append("total_context_reserve_exceeded")
    return {
        "task_id": str(task_id),
        "initial_prompt_tokens": token_count,
        "initial_prompt_limit": prompt_limit,
        "reserved_tokens": reserve,
        "prompt_plus_reserve_tokens": token_count + reserve,
        "max_context_tokens": budget.max_context_tokens,
        "violations": violations,
        "status": "rejected" if violations else "ok",
    }


def require_initial_prompt_fits(measurement):
    if measurement["violations"]:
        raise ValueError(
            f"task {measurement['task_id']}: full initial prompt has "
            f"{measurement['initial_prompt_tokens']} tokens (limit "
            f"{measurement['initial_prompt_limit']}); prompt + reserves = "
            f"{measurement['prompt_plus_reserve_tokens']} / "
            f"{measurement['max_context_tokens']}; "
            + ", ".join(measurement["violations"])
        )


def summarize_initial_prompts(rows, expected_task_ids):
    """Nearest-rank quantiles; unmeasured/reset-failed tasks remain explicit."""
    values = sorted(
        row["initial_prompt_tokens"] for row in rows if "initial_prompt_tokens" in row
    )
    measured = {row["task_id"] for row in rows if "initial_prompt_tokens" in row}
    failed = [row for row in rows if row.get("status") != "ok"]
    missing = sorted(set(map(str, expected_task_ids)) - measured)
    return {
        "measurement": "actual_runtime_initial_prompt_with_policy_tools_and_confirmation",
        "quantile_method": "nearest_rank",
        "measured_samples": len(values),
        "samples_per_task": dict(Counter(row["task_id"] for row in rows)),
        "missing_task_ids": missing,
        "all_requested_samples_passed": bool(values) and not failed and not missing,
        "tokens": {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            **{
                f"p{q}": values[max(0, ceil(q / 100 * len(values)) - 1)]
                if values
                else None
                for q in (50, 90, 95, 99)
            },
        },
        "rows": rows,
    }
