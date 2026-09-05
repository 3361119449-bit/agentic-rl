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
Ask the user to reply with an unqualified "Yes" (or "Yes, proceed"). A reply
that adds conditions or changes the requested details requires a new proposal.
""".strip()

ACTION_PROPOSAL_RE = re.compile(
    r"<action_proposal>\s*(\{.*?\})\s*</action_proposal>", re.DOTALL
)
TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def validate_tool_turn(raw: str, parsed_count: int) -> str | None:
    """Check the entire Qwen turn, including blocks Hermes silently discarded."""
    starts, ends = raw.count("<tool_call>"), raw.count("</tool_call>")
    if not starts and not ends and not parsed_count:
        return None
    blocks = TOOL_BLOCK_RE.findall(raw)
    if starts != ends or starts != len(blocks) or starts != parsed_count:
        return "parse_error"
    if starts != 1:
        return "multiple_tool_calls"
    outside = TOOL_BLOCK_RE.sub("", raw).strip()
    # Only strip one terminal assistant EOS, never tokens embedded in text.
    if outside.endswith("<|im_end|>"):
        outside = outside[: -len("<|im_end|>")].strip()
    return "mixed_content_and_tool_call" if outside else None


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
        key=lambda item: tuple(map(str, item.absolute_path)),
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

    @staticmethod
    def _affirmative(content: Any) -> bool:
        # A prefix match also accepts "Yes, but cancel B instead", authorizing
        # the wrong payload. Keep the protocol deliberately conservative.
        normalized = re.sub(r"[,.!]", " ", str(content).lower())
        normalized = " ".join(normalized.split())
        return (
            re.fullmatch(
                r"yes(?: please|(?: (?:please )?(?:proceed|go ahead|cancel it|book it|do it)(?: please)?))?",
                normalized,
            )
            is not None
        )

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

    def observe_visible_assistant_text(self, content: str, turn_id: int) -> None:
        """Call only AFTER a pure-text turn was successfully delivered to Tau2."""
        self.pending = None
        if "<tool_call>" in content or "</tool_call>" in content:
            return
        matches = ACTION_PROPOSAL_RE.findall(content)
        if len(matches) != 1 or content.count("<action_proposal>") != 1:
            return
        try:
            parsed = json.loads(matches[0])
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
            return
        name, arguments = parsed["name"], parsed["arguments"]
        if (
            isinstance(name, str)
            and name in MUTATING_TOOLS
            and isinstance(arguments, dict)
        ):
            self._set_proposal(name, arguments, turn_id)

    def observe_user_reply(self, content: str, turn_id: int) -> None:
        """A new non-affirmative reply revokes any earlier authorization."""
        if (
            self.pending is None
            or self.pending.consumed
            or turn_id <= self.pending.proposal_turn_id
        ):
            return
        self.pending.confirmation_turn_id = (
            turn_id if self._affirmative(content) else None
        )

    def authorize(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, ActionProposal | None]:
        requested_hash = action_hash(name, arguments)
        if (
            self.pending is not None
            and self.pending.proposal_hash == requested_hash
            and self.pending.confirmation_turn_id is not None
            and not self.pending.consumed
        ):
            return True, self.pending
        return False, None

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
