"""Pinned Tau2 verifiers and actual airline DB, without simulators or API calls."""

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from tau2_agentic_rl.reward.progress import build_progress_trace, evaluate_tau2_prefix


def test_real_prefix_verifiers_and_initial_state_isolation():
    tau2 = pytest.importorskip("tau2")
    from tau2.registry import registry

    sha = subprocess.check_output(
        ["git", "-C", str(Path(tau2.__file__).parent), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
    tasks = [
        task.model_dump(mode="json") for task in registry.get_tasks_loader("airline")()
    ]
    task = deepcopy(
        next(t for t in tasks if "DB" in t["evaluation_criteria"]["reward_basis"])
    )
    task["evaluation_criteria"]["communicate_info"] = ["PROGRESS_CONTRACT_SENTINEL"]
    initial = task.get("initial_state") or {}
    history = initial.get("message_history") or []
    messages = history + [
        {"role": "user", "content": "PROGRESS_CONTRACT_SENTINEL"},
        {"role": "assistant", "content": "PROGRESS_CONTRACT_SENTINEL"},
    ]
    before = deepcopy((task, messages))
    first = evaluate_tau2_prefix(task, messages[:-1])
    last = evaluate_tau2_prefix(task, messages)
    assert type(first["db"]) is bool
    assert first["db"] == last["db"]
    assert first["communicate"] == [False]
    assert last["communicate"] == [True]
    trace = build_progress_trace(
        task=task,
        messages=messages,
        prefix_lengths=[len(messages) - 1, len(messages)],
        events=[],
        required_actions=[],
        dependencies=[],
        transfer_rule={},
        initial_state_fingerprint="contract-initial",
    )
    assert trace["phi"] == [0, 1 if first["db"] else 0.5]
    assert (task, messages) == before
