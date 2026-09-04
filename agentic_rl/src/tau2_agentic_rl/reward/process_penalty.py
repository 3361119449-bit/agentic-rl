"""Deduplicated process penalties from the training plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from tau2_agentic_rl.schemas import ProcessPenaltyResult, ToolEvent

BASE_ERROR_PRECEDENCE = (
    "parse_error",
    "unknown_tool",
    "schema_invalid",
    "model_caused_execution_error",
    "confirmation_required",
    "multiple_tool_calls",
)


@dataclass(frozen=True)
class ProcessPenaltyConfig:
    """First-version process penalty values."""

    penalties: dict[str, float] = field(
        default_factory=lambda: {
            "parse_error": 0.10,
            "unknown_tool": 0.10,
            "schema_invalid": 0.08,
            "model_caused_execution_error": 0.08,
            "confirmation_required": 0.08,
            "multiple_tool_calls": 0.10,
            "unchanged_retry": 0.06,
            "duplicate_no_progress": 0.03,
            "extra_assistant_turn": 0.02,
        }
    )
    cap: float = 0.20
    soft_turn_limit: int = 15
    over_turn_cap: float = 0.08


def compute_process_penalty(
    events: list[ToolEvent],
    assistant_turns: int,
    config: ProcessPenaltyConfig | None = None,
) -> ProcessPenaltyResult:
    """Apply one base error per call, then independent retry/repetition errors."""
    config = config or ProcessPenaltyConfig()
    penalty_events: list[dict] = []
    total = 0.0

    for event in events:
        if event.error_kind:
            kind = next(
                item for item in BASE_ERROR_PRECEDENCE if item == event.error_kind
            )
            amount = config.penalties[kind]
            total += amount
            penalty_events.append(
                {"event_id": event.event_id, "kind": kind, "penalty": amount}
            )
        if event.unchanged_retry:
            amount = config.penalties["unchanged_retry"]
            total += amount
            penalty_events.append(
                {
                    "event_id": event.event_id,
                    "kind": "unchanged_retry",
                    "penalty": amount,
                }
            )
        if event.no_progress:
            amount = config.penalties["duplicate_no_progress"]
            total += amount
            penalty_events.append(
                {
                    "event_id": event.event_id,
                    "kind": "duplicate_no_progress",
                    "penalty": amount,
                }
            )

    extra_turns = max(0, assistant_turns - config.soft_turn_limit)
    turn_penalty = min(
        extra_turns * config.penalties["extra_assistant_turn"],
        config.over_turn_cap,
    )
    if turn_penalty:
        total += turn_penalty
        penalty_events.append(
            {
                "kind": "over_soft_turn_limit",
                "count": extra_turns,
                "penalty": turn_penalty,
            }
        )

    total = min(total, config.cap)
    return ProcessPenaltyResult(
        penalty=total,
        process_reward=1.0 - total,
        events=penalty_events,
    )
