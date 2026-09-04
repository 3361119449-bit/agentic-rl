"""Deterministic validation, confirmation binding, and observation bounds."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS
from tau2_agentic_rl.versions import sha256_json

CONFIRMATION_PROTOCOL = """
Database-write confirmation protocol:
Before calling any database-mutating tool, describe the exact operation to the
user and include the exact tool payload in
<action_proposal>{"name":"...","arguments":{...}}</action_proposal>. Ask for
confirmation. A later affirmative user message authorizes exactly that name and
argument object once. Any argument change requires a new proposal and a new
confirmation. Never put the proposal tag inside a tool call.
""".strip()

ACTION_PROPOSAL_RE = re.compile(
    r"<action_proposal>\s*(\{.*?\})\s*</action_proposal>", re.DOTALL
)


@dataclass(frozen=True)
class ToolValidation:
    name: str
    arguments: dict[str, Any]
    error_kind: str | None = None
    detail: str | None = None

    @property
    def valid(self) -> bool:
        return self.error_kind is None


def validate_tool_call(
    name: str,
    raw_arguments: str,
    schemas_by_name: dict[str, dict[str, Any]],
    tool_names: set[str],
) -> ToolValidation:
    """Validate untrusted model output completely before touching Tau2."""
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return ToolValidation(name, {}, "schema_invalid", f"invalid JSON: {exc}")
    if not isinstance(arguments, dict):
        return ToolValidation(name, {}, "schema_invalid", "arguments must be an object")
    if name not in tool_names or name not in schemas_by_name:
        return ToolValidation(name, arguments, "unknown_tool", f"unknown tool: {name}")
    errors = sorted(
        Draft202012Validator(schemas_by_name[name]).iter_errors(arguments),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.absolute_path)) or "arguments"
        return ToolValidation(
            name,
            arguments,
            "schema_invalid",
            f"{location}: {first.message}",
        )
    return ToolValidation(name, arguments)


async def execute_validated_tool_call(validation: ToolValidation, executor: Any) -> Any:
    """Make it impossible for an invalid call to reach a supplied backend."""
    if not validation.valid:
        raise ValueError(
            f"refusing backend execution: {validation.error_kind}: {validation.detail}"
        )
    return await executor(validation.name, validation.arguments)


def synthetic_tool_error(name: str, error_kind: str, detail: str) -> dict[str, Any]:
    """Create a short local observation without calling the Tau2 backend."""
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps(
            {"error": True, "type": error_kind, "message": detail},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "error": True,
    }


def action_hash(name: str, arguments: dict[str, Any]) -> str:
    """Bind confirmation to the exact canonical tool name and arguments."""
    return sha256_json({"name": name, "arguments": arguments})


@dataclass
class ActionProposal:
    name: str
    arguments: dict[str, Any]
    proposal_hash: str
    proposal_turn_id: int
    confirmation_turn_id: int | None = None
    consumed: bool = False


class ConfirmationTracker:
    """One-shot, argument-bound confirmation state for database writes."""

    def __init__(self) -> None:
        self.pending: ActionProposal | None = None
        self._observed_messages = 0

    @staticmethod
    def _affirmative(content: Any) -> bool:
        return re.match(r"^yes\b", str(content).strip(), re.IGNORECASE) is not None

    def _set_proposal(
        self, name: str, arguments: dict[str, Any], turn_id: int
    ) -> ActionProposal:
        self.pending = ActionProposal(
            name=name,
            arguments=copy.deepcopy(arguments),
            proposal_hash=action_hash(name, arguments),
            proposal_turn_id=turn_id,
        )
        return self.pending

    def observe_messages(self, messages: list[dict[str, Any]]) -> None:
        for index, message in enumerate(
            messages[self._observed_messages :], self._observed_messages
        ):
            role = message.get("role")
            if role == "assistant":
                content = str(message.get("content", ""))
                matches = ACTION_PROPOSAL_RE.findall(content)
                for raw in matches:
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    name = parsed.get("name") if isinstance(parsed, dict) else None
                    arguments = (
                        parsed.get("arguments") if isinstance(parsed, dict) else None
                    )
                    if name in MUTATING_TOOLS and isinstance(arguments, dict):
                        self._set_proposal(
                            str(name), arguments, int(message.get("turn_idx", index))
                        )
            elif (
                role == "user"
                and self.pending is not None
                and not self.pending.consumed
                and self._affirmative(message.get("content", ""))
            ):
                self.pending.confirmation_turn_id = int(message.get("turn_idx", index))
        self._observed_messages = len(messages)

    def authorize(
        self,
        name: str,
        arguments: dict[str, Any],
        turn_id: int,
        messages: list[dict[str, Any]],
    ) -> tuple[bool, ActionProposal]:
        self.observe_messages(messages)
        requested_hash = action_hash(name, arguments)
        if (
            self.pending is not None
            and self.pending.proposal_hash == requested_hash
            and self.pending.confirmation_turn_id is not None
            and not self.pending.consumed
        ):
            return True, self.pending
        return False, self._set_proposal(name, arguments, turn_id)

    def consume(self, proposal_hash: str) -> None:
        if self.pending is None or self.pending.proposal_hash != proposal_hash:
            raise ValueError("cannot consume a different confirmation proposal")
        self.pending.consumed = True


def truncate_message_contents(
    messages: list[dict[str, Any]], tokenizer: Any, max_content_tokens: int
) -> tuple[list[dict[str, Any]], bool]:
    """Deterministically cap environment content before it enters the prompt."""
    if max_content_tokens < 0:
        raise ValueError("max_content_tokens must be non-negative")
    result = copy.deepcopy(messages)
    tokenized = [
        tokenizer.encode(str(item.get("content", "")), add_special_tokens=False)
        for item in result
    ]
    total = sum(map(len, tokenized))
    if total <= max_content_tokens:
        return result, False

    remaining = max_content_tokens
    marker = " [OBSERVATION_TRUNCATED]"
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    for item, content_ids in zip(result, tokenized, strict=True):
        keep = min(len(content_ids), remaining)
        if keep < len(content_ids) and keep > len(marker_ids):
            kept_ids = content_ids[: keep - len(marker_ids)] + marker_ids
        else:
            kept_ids = content_ids[:keep]
        item["content"] = tokenizer.decode(kept_ids, skip_special_tokens=True)
        if keep < len(content_ids):
            item["observation_truncated"] = True
        remaining -= keep
    return result, True
