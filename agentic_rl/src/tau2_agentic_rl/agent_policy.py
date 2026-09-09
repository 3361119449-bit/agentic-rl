"""The exact AReaL SFT system message, never a Tau2-policy fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_agent_system_prompt(project: dict, root: Path) -> str:
    binding = project.get("agent_policy") or {}
    if not binding.get("path") or not binding.get("system_prompt_sha256"):
        raise ValueError(
            "missing SFT agent_policy binding; no official-policy fallback"
        )
    artifact = json.loads((root / binding["path"]).read_text(encoding="utf-8"))
    prompt = artifact.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SFT system prompt must be nonempty text")
    digest = prompt_sha256(prompt)
    if digest != binding["system_prompt_sha256"] or digest != artifact.get(
        "system_prompt_sha256"
    ):
        raise ValueError("SFT system prompt hash mismatch")
    extract_airline_policy(prompt)  # Fail before a live reset/API call.
    return prompt


def extract_airline_policy(system_prompt: str) -> str:
    """Give the Judge the SFT policy, not the agent's output-format instructions."""
    start, end = "\n<policy>\n", "\n</policy>"
    # Instructions also mention the literal '<policy>' inline. Only the
    # standalone block delimiters identify the policy supplied to the Judge.
    if system_prompt.count(start) != 1 or system_prompt.count(end) != 1:
        raise ValueError("SFT system prompt must contain exactly one policy block")
    policy = system_prompt.split(start, 1)[1].split(end, 1)[0]
    if not policy.strip():
        raise ValueError("SFT policy block is empty")
    return policy
