#!/usr/bin/env python3
"""Convert successful official Tau2 train rollouts into prefix/answer SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# Official airline split in tau2-bench v1.0.1 (commit a2c024725189...).
OFFICIAL_TRAIN_IDS = {
    "0", "1", "3", "4", "5", "7", "9", "10", "11", "12", "14", "15",
    "17", "20", "21", "23", "27", "28", "33", "34", "36", "38", "39",
    "40", "41", "42", "43", "46", "47", "49",
}
SUCCESS_TOLERANCE = 1e-6
ALLOWED_TERMINATIONS = {"user_stop", "agent_stop"}
FORBIDDEN_REASONING_KEYS = {"thinking", "reasoning", "reasoning_content"}
THINK_TAG_RE = re.compile(r"</?think(?:\s[^>]*)?>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-successes-per-task", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--skip-token-count",
        action="store_true",
        help="For offline structural tests only; production conversion should count tokens.",
    )
    parser.add_argument("--allow-tool-errors", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve()
    metadata_path = path / "results.json" if path.is_dir() else path
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Tau2 results not found: {metadata_path}")
    root = load_json(metadata_path)
    if not isinstance(root, dict):
        raise ValueError("Tau2 results root must be a JSON object")
    simulations = root.get("simulations")
    if isinstance(simulations, list) and simulations:
        return root, simulations

    simulations_dir = metadata_path.parent / "simulations"
    if not simulations_dir.is_dir():
        return root, simulations if isinstance(simulations, list) else []
    loaded = []
    for simulation_path in sorted(simulations_dir.glob("*.json")):
        simulation = load_json(simulation_path)
        if not isinstance(simulation, dict):
            raise ValueError(f"Invalid simulation object: {simulation_path}")
        loaded.append(simulation)
    return root, loaded


def has_nonempty_reasoning(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_REASONING_KEYS and child not in (
                None,
                "",
                [],
                {},
            ):
                return True
            if has_nonempty_reasoning(child):
                return True
    elif isinstance(value, list):
        return any(has_nonempty_reasoning(child) for child in value)
    elif isinstance(value, str):
        return bool(THINK_TAG_RE.search(value))
    return False


def normalize_tool_calls(value: Any) -> list[dict[str, str]] | None:
    if not value:
        return None
    if not isinstance(value, list):
        raise ValueError("tool_calls must be a list")
    calls: list[dict[str, str]] = []
    for call in value:
        if not isinstance(call, dict):
            raise ValueError("tool call must be an object")
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("tool call is missing its function name")
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        calls.append({"name": name, "arguments": arguments})
    return calls


def normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("message must be an object")
    role = message.get("role")
    if role not in {"user", "assistant", "tool"}:
        raise ValueError(f"unsupported trajectory role: {role!r}")
    content = message.get("content", "")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    if THINK_TAG_RE.search(content):
        raise ValueError("visible message contains a <think> tag")
    return {
        "role": role,
        "content": content,
        "name": None,
        "tool_calls": normalize_tool_calls(message.get("tool_calls")),
    }


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content, "name": None, "tool_calls": None}


def is_success(simulation: dict[str, Any]) -> bool:
    reward_info = simulation.get("reward_info")
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    return isinstance(reward, (int, float)) and math.isclose(
        float(reward), 1.0, abs_tol=SUCCESS_TOLERANCE
    )


def has_tool_error(messages: Iterable[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "tool" and message.get("error") is True
        for message in messages
    )


def load_tokenizer(model: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("install transformers to enforce the 16K limit") from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=False)


def token_length(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    return len(encoded)


def stable_fingerprint(messages: list[dict[str, Any]], answer: dict[str, Any]) -> str:
    payload = json.dumps(
        {"messages": messages, "answer": answer},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_sort_key(simulation: dict[str, Any]) -> tuple[int, int, str]:
    task_id = str(simulation.get("task_id", ""))
    numeric_task = int(task_id) if task_id.isdigit() else 10**9
    trial = simulation.get("trial")
    return numeric_task, int(trial) if isinstance(trial, int) else 10**9, str(
        simulation.get("id", "")
    )


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("wb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            handle.write(raw)
            digest.update(raw)
            count += 1
    os.replace(temporary, path)
    return count, digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_successes_per_task < 0 or args.max_tokens < 1:
        raise ValueError("caps must be non-negative and max tokens must be positive")
    root, simulations = load_results(args.results)
    if not simulations:
        raise ValueError("results contain no simulations")

    info = root.get("info") if isinstance(root.get("info"), dict) else {}
    environment_info = (
        info.get("environment_info")
        if isinstance(info.get("environment_info"), dict)
        else {}
    )
    if environment_info.get("domain_name") != "airline":
        raise ValueError("only Tau2 airline results are accepted")

    observed_ids = {str(simulation.get("task_id")) for simulation in simulations}
    forbidden_ids = observed_ids - OFFICIAL_TRAIN_IDS
    if forbidden_ids:
        raise ValueError(
            "results contain IDs outside the official airline train split: "
            + ", ".join(sorted(forbidden_ids))
        )

    context = load_json(args.context.resolve())
    if not isinstance(context, dict) or context.get("domain") != "airline":
        raise ValueError("context must be an exported Tau2 airline context")
    prompt = context.get("system_prompt")
    tools = context.get("tools")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("context is missing system_prompt")
    if not isinstance(tools, list) or not tools:
        raise ValueError("context is missing tool schemas")
    results_commit = info.get("git_commit")
    context_commit = context.get("tau2_git_commit")
    if results_commit and context_commit and results_commit != context_commit:
        raise ValueError(
            f"results/context Tau2 commit mismatch: {results_commit} != {context_commit}"
        )

    tokenizer = None if args.skip_token_count else load_tokenizer(args.tokenizer)
    counters: Counter[str] = Counter()
    accepted_per_task: defaultdict[str, int] = defaultdict(int)
    fingerprints: set[str] = set()
    rows: list[dict[str, Any]] = []

    for simulation in sorted(simulations, key=task_sort_key):
        task_id = str(simulation.get("task_id"))
        if not is_success(simulation):
            counters["rejected_reward"] += 1
            continue
        termination = str(simulation.get("termination_reason", ""))
        if termination not in ALLOWED_TERMINATIONS:
            counters["rejected_termination"] += 1
            continue
        source_messages = simulation.get("messages")
        if not isinstance(source_messages, list) or not source_messages:
            counters["rejected_missing_messages"] += 1
            continue
        if has_nonempty_reasoning(source_messages):
            counters["rejected_reasoning"] += 1
            continue
        if not args.allow_tool_errors and has_tool_error(source_messages):
            counters["rejected_tool_error"] += 1
            continue
        cap = args.max_successes_per_task
        if cap and accepted_per_task.get(task_id, 0) >= cap:
            counters["rejected_task_cap"] += 1
            continue

        try:
            normalized = [normalize_message(message) for message in source_messages]
        except ValueError:
            counters["rejected_invalid_message"] += 1
            continue

        simulation_rows: list[dict[str, Any]] = []
        history = [system_message(prompt)]
        simulation_id = str(simulation.get("id", ""))
        for turn_index, message in enumerate(normalized):
            if message["role"] == "assistant" and history[-1]["role"] in {"user", "tool"}:
                if message["content"].strip() or message["tool_calls"]:
                    fingerprint = stable_fingerprint(history, message)
                    if fingerprint not in fingerprints:
                        full_messages = history + [message]
                        length = (
                            None
                            if tokenizer is None
                            else token_length(tokenizer, full_messages, tools)
                        )
                        if length is not None and length > args.max_tokens:
                            counters["rejected_over_16k_rows"] += 1
                        else:
                            row = {
                                "messages": list(history),
                                "answer": message,
                                "tools": tools,
                                "metadata": {
                                    "source_dialog_id": f"tau2_train_{task_id}_{simulation_id}",
                                    "turn_index": turn_index,
                                    "tau2_task_id": task_id,
                                    "tau2_simulation_id": simulation_id,
                                    "trial": simulation.get("trial"),
                                    "reward": 1.0,
                                    "termination_reason": termination,
                                    "agent_model": (
                                        info.get("agent_info", {}).get("llm")
                                        if isinstance(info.get("agent_info"), dict)
                                        else None
                                    ),
                                    "user_model": (
                                        info.get("user_info", {}).get("llm")
                                        if isinstance(info.get("user_info"), dict)
                                        else None
                                    ),
                                    "tau2_git_commit": info.get("git_commit"),
                                    "token_count": length,
                                    "correct": 1,
                                },
                            }
                            if has_nonempty_reasoning(row):
                                raise AssertionError("reasoning leaked into an output row")
                            fingerprints.add(fingerprint)
                            simulation_rows.append(row)
            history.append(message)

        if simulation_rows:
            rows.extend(simulation_rows)
            accepted_per_task[task_id] += 1
            counters["accepted_simulations"] += 1
            counters["accepted_rows"] += len(simulation_rows)
        else:
            counters["rejected_no_usable_targets"] += 1

    if not rows:
        raise RuntimeError("no SFT rows survived conversion")

    row_count, output_sha256 = write_jsonl_atomic(args.output, rows)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else args.output.resolve().with_suffix(".manifest.json")
    )
    manifest = {
        "input_results": str(args.results.resolve()),
        "context": str(args.context.resolve()),
        "output": str(args.output.resolve()),
        "output_rows": row_count,
        "output_sha256": output_sha256,
        "accepted_tasks": sorted(accepted_per_task, key=lambda value: int(value)),
        "accepted_successful_simulations": sum(accepted_per_task.values()),
        "counters": dict(sorted(counters.items())),
        "max_successes_per_task": args.max_successes_per_task,
        "tokenizer": None if args.skip_token_count else args.tokenizer,
        "max_tokens": None if args.skip_token_count else args.max_tokens,
        "loss_targets": "each agent assistant turn after a user/tool message",
        "copied_hidden_task_or_evaluation_fields": False,
        "reasoning_thinking_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
