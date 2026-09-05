import asyncio
import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.base_identity import (
    capture_base_identity,
    save_base_identity,
    validate_adapter_base,
)
from tau2_agentic_rl.concurrency import BudgetState, SharedBudget
from tau2_agentic_rl.evaluation import evaluation_coverage, initialize_evaluation
from tau2_agentic_rl.reward.mandatory_policy import (
    evaluate_mandatory_policy,
    evaluate_task_safety,
)
from tau2_agentic_rl.reward.required_actions import evaluate_required_actions
from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.rollout_audit import audited_official_scores, snapshot_transcript
from tau2_agentic_rl.schemas import (
    JudgeResult,
    OfficialScores,
    ToolEvent,
    TrajectoryRecord,
)
from tau2_agentic_rl.scoring_retry import retry_scoring
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.versions import sha256_json


@pytest.mark.parametrize(
    "reason", ["generation_truncated", "hard_turn_limit", "budget_exhausted"]
)
def test_external_stop_keeps_progress_but_not_official_success(reason):
    official = audited_official_scores(
        1.0,
        {
            "reward_basis": ["DB"],
            "db_check": {"db_match": True, "db_reward": 1.0},
        },
        reason,
    )
    custom = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=official,
        judge=JudgeResult(),
    )
    assert official.reward == 0 and official.db_score == 1
    assert custom.train_reward == 1  # Existing partial/completion formula is unchanged.


def test_full_interaction_snapshot_excludes_unsent_and_later_cleanup():
    full = [
        {"role": "user", "content": "refund?"},
        {"role": "tool", "content": "X" * 20000},
    ]
    env = SimpleNamespace(full_trajectory=lambda: full)
    actor_context = [
        *full[:1],
        {"role": "tool", "content": "[truncated]"},
        {"role": "assistant", "content": "Unsent refund explanation"},
    ]
    snapshot = snapshot_transcript(env)
    full.append({"role": "assistant", "content": "###STOP###"})
    assert len(snapshot) == 2 and len(snapshot[1]["content"]) == 20000
    assert not any(m["role"] == "assistant" for m in snapshot)
    assert actor_context[-1]["content"] not in json.dumps(snapshot)


def test_failed_partial_write_blocks_safety_without_completing_action():
    event = ToolEvent(
        event_id="partial",
        sequence=0,
        turn_id=1,
        name="cancel_reservation",
        arguments={"reservation_id": "ABC"},
        success=False,
        db_effect=True,
    )
    assert not evaluate_task_safety([event], []).passed
    checks = {c.rule_id: c for c in evaluate_mandatory_policy([event], [])}
    assert not checks["failed_tool_with_database_mutation"].passed
    assert not checks["confirmation_before_database_write"].passed
    required = [{"action_id": "a", "name": event.name, "arguments": event.arguments}]
    assert evaluate_required_actions(required, [event]).component.value == 0


def test_shared_async_limits_and_release_on_exception():
    async def run():
        limits = {"trajectories": 2, "user_api": 1, "judge_api": 1}
        # Two independently created clients must resolve the same budget.
        clients = [SharedBudget(limits), SharedBudget(limits)]

        async def trajectory(i):
            client = clients[i % 2]
            async with client.aslot("trajectories"):
                async with client.aslot("user_api"):
                    await asyncio.sleep(0.005)
                async with client.aslot("judge_api"):
                    await asyncio.sleep(0.005)
                if i == 1:
                    raise RuntimeError("fixture")

        results = await asyncio.gather(
            *(trajectory(i) for i in range(16)), return_exceptions=True
        )
        assert isinstance(results[1], RuntimeError)
        metrics = clients[0].call("snapshot")
        assert metrics["peak_active_trajectories"] == 2
        assert (
            metrics["peak_user_api_inflight"] == metrics["peak_judge_api_inflight"] == 1
        )
        assert set(metrics["active"].values()) == {0}

    asyncio.run(run())


def test_budget_rejects_worker_config_drift():
    limits = {"trajectories": 2, "user_api": 1, "judge_api": 1}
    state = BudgetState(limits)
    with pytest.raises(ValueError, match="disagree"):
        state.try_acquire("trajectories", "a", {**limits, "trajectories": 3})


def test_adapter_rejects_wrong_base_before_loading_weights(scratch_dir):
    base = scratch_dir / "base"
    base.mkdir()
    for name, content in {
        "config.json": "{}",
        "tokenizer_config.json": '{"chat_template": "A"}',
        "tokenizer.json": "{}",
        "model.safetensors": "fixture",
    }.items():
        (base / name).write_text(content, encoding="utf-8")
    adapter = scratch_dir / "adapter"
    identity = capture_base_identity(base)
    save_base_identity(adapter, identity)
    assert validate_adapter_base(adapter)[0] == base.resolve()
    moved = scratch_dir / "moved"
    shutil.copytree(base, moved)
    assert validate_adapter_base(adapter, moved)[0] == moved.resolve()
    (moved / "model.safetensors").write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_adapter_base(adapter, moved)
    (base / "tokenizer_config.json").write_text(
        '{"chat_template": "B"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_adapter_base(adapter)


def scoring_failure_record():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "delivered"},
    ]
    frozen = {
        "judge": {
            "task": {},
            "policy": "policy",
            "trajectory": {
                "messages": messages,
                "tool_events": [],
                "termination_reason": "user_stop",
            },
            "semantic_checks": [],
            "mandatory_policy_checks": [],
            "transfer_rule": {},
        },
        "required_actions": [],
        "official_scores": OfficialScores(reward=1).model_dump(),
        "action_dependencies": [],
        "reward_project_config": {},
    }
    return TrajectoryRecord(
        trajectory_id="same-id",
        task_id="0",
        split="internal_dev",
        policy_version=0,
        annotation_version="v1",
        reward_version="v1",
        environment_seed=42,
        termination_reason="user_stop",
        assistant_turns=1,
        trajectory_tokens=10,
        environment_transcript=messages,
        messages=deepcopy(messages),
        scoring_inputs=frozen,
        official_scores=OfficialScores(reward=1),
        metadata={
            "failure_phase": "judge",
            "evaluation_sample_index": 0,
            "scoring_inputs_sha256": sha256_json(frozen),
        },
    )


def test_judge_retry_preserves_interaction_and_never_refills_its_slot(scratch_dir):
    manifest = initialize_evaluation(
        scratch_dir / "eval",
        {
            "task_ids": ["0"],
            "samples_per_task": 4,
            "split": "internal_dev",
            "record_split": "internal_dev",
        },
        resume=False,
    )
    store = TrajectoryStore(
        scratch_dir / "eval/trajectories", attach_evaluation_identity=False
    )
    record = scoring_failure_record()
    record.metadata["evaluation_manifest_id"] = manifest["manifest_id"]
    store.save(record)
    before = record.model_dump()
    coverage = evaluation_coverage(store.root, manifest)
    assert len(coverage["scoring_pending_records"]) == 1
    assert {x["sample_index"] for x in coverage["missing_slots"]} == {1, 2, 3}

    class Judge:
        async def evaluate(self, **inputs):
            assert inputs == before["scoring_inputs"]["judge"]
            return JudgeResult(), "raw", "prompt", "cache"

    assert asyncio.run(retry_scoring(record, Judge(), store))
    after = next(store.records()).model_dump()
    for field in (
        "trajectory_id",
        "environment_seed",
        "termination_reason",
        "official_scores",
        "messages",
        "environment_transcript",
        "scoring_inputs",
        "tool_events",
        "token_turns",
    ):
        assert after[field] == before[field]
    assert len(list(store.root.glob("*.json"))) == 1
    assert not evaluation_coverage(store.root, manifest)["scoring_pending_records"]


def test_judge_retry_failure_keeps_slot_pending(scratch_dir):
    record = scoring_failure_record()

    class Judge:
        async def evaluate(self, **inputs):
            raise RuntimeError("judge unavailable")

    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    assert not asyncio.run(retry_scoring(record, Judge(), store))
    assert record.termination_reason == "user_stop"
    assert record.official_scores.reward == 1
    assert record.custom_reward is None
    assert len(record.metadata["scoring_retries"]) == 1


def test_judge_retry_rejects_modified_official_score(scratch_dir):
    record = scoring_failure_record()
    record.official_scores.reward = 0
    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    with pytest.raises(ValueError, match="interaction differs"):
        asyncio.run(retry_scoring(record, None, store))


def test_cancelling_queued_trajectory_does_not_leak_or_release_another_lease():
    async def run():
        budget = SharedBudget({"trajectories": 1, "user_api": 2, "judge_api": 2})
        async with budget.aslot("trajectories"):

            async def queued():
                async with budget.aslot("trajectories"):
                    pytest.fail("queue cancellation should not enter the environment")

            task = asyncio.create_task(queued())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert budget.call("snapshot")["active"]["trajectories"] == 1
        assert budget.call("snapshot")["active"]["trajectories"] == 0

    asyncio.run(run())
