import asyncio
import json

import httpx
import pytest

from test_procredit_rollout import rollout_fixture
from test_routed_judge import answer
from test_trajectory_once_judge import mock_client
from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.concurrency import SharedBudget
from tau2_agentic_rl.full_official_group import OfficialGroup, choose_group_branch
from tau2_agentic_rl.procredit_runtime import queue_group_reports


@pytest.mark.parametrize("scores,expected", [
    ([1.] * 8, True), ([1.] * 7 + [0.], False), ([1.] * 7 + [None], False),
])
def test_group_branch(scores, expected):
    async def run():
        group = OfficialGroup("group1")
        async def member(i):
            if scores[i] is None:
                return
            decision = await choose_group_branch(scores[i], f"id{i}")
            assert bool(decision) == expected
        await asyncio.wait_for(asyncio.gather(*[
            group.session(i, member(i)) for i in range(8)
        ]), 2)
    asyncio.run(run())


def test_all_pass_can_settle_with_one_interaction_lease():
    async def run():
        group = OfficialGroup("group1")
        budget = SharedBudget({"trajectories": 1, "user_api": 1, "judge_api": 1})
        async def member(i):
            async with budget.aslot("trajectories") as lease:
                decision = await choose_group_branch(1., f"id{i}", lease)
                assert decision["members"] == [f"id{k}" for k in range(8)]
        await asyncio.wait_for(asyncio.gather(*[
            group.session(i, member(i)) for i in range(8)
        ]), 4)
    asyncio.run(run())


@pytest.mark.parametrize("missing", [False, True])
def test_mixed_or_exhausted_group_keeps_real_scoring_pipeline(monkeypatch, scratch_dir, missing):
    requests = []
    def respond(request):
        body = json.loads(json.loads(request.content)["messages"][1]["content"])
        requests.append(body)
        wire = answer()
        wire["turns"] = [{"message_index": item["message_index"], "verdict": "neutral"}
                         for item in body["targets"]]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(wire)}}]})

    client = mock_client(monkeypatch, scratch_dir / "cache", respond)
    async def run():
        group = OfficialGroup("group1")
        async def member(i):
            if missing and i == 7:
                raise RuntimeError("exhausted slot")
            loop, scope = rollout_fixture(
                scratch_dir / str(i), credit_version="procredit-turn-v4",
                task_score_mode="official_semantic_progress_v1", final_turn_rule="simple-turn-v2",
                judge_mode="trajectory_routed_v1", actual_judge=client,
                routing_rule={"task_id": "0", "mode": "always", "hints": []}, natural_stop_at=1,
                prepare_only=True,
            )
            loop.project["reward"]["full_official_group_terminal_only"] = True
            if i == 7:
                scope["Tau2GymAdapter"].official_reward_payload = lambda self: (
                    0., {"reward_basis": ["COMMUNICATE"], "communicate_checks": [{"met": False}]}
                )
            output = await loop._run_trajectory({}, trajectory_id=f"id{i}", extra_info={"task_id": "0"})
            record = next(loop.store.records())
            assert record.judge_result is not None and record.metadata["simple_turn_labels"]["judge_calls"]
            assert "official_group" not in output.extra_fields["procredit"]
            return output.extra_fields
        results = await asyncio.wait_for(asyncio.gather(*[
            group.session(i, member(i)) for i in range(8)
        ], return_exceptions=True), 20)
        for item in results[:7 if missing else 8]:
            if isinstance(item, BaseException):
                raise item
        if missing:
            assert str(results[-1]) == "exhausted slot"
        assert requests
    asyncio.run(run())


@pytest.mark.parametrize("different", [False, True])
def test_complete_rollouts_skip_judge_and_use_only_scalar_credit(scratch_dir, different):
    async def run():
        group = OfficialGroup("group1")
        async def member(i):
            path = scratch_dir / str(i)
            path.mkdir()
            loop, scope = rollout_fixture(
                path, credit_version="procredit-turn-v4",
                task_score_mode="official_semantic_progress_v1", final_turn_rule="simple-turn-v2",
                judge_mode="trajectory_routed_v1",
                routing_rule={"task_id": "0", "mode": "always", "hints": []},
                natural_stop_at=1, prepare_only=True,
            )
            loop.project["reward"]["full_official_group_terminal_only"] = True
            if different and i == 7:
                from tau2_agentic_rl.reward.progress import build_progress_trace
                scope["build_progress_trace"] = lambda **kw: build_progress_trace(
                    **kw, evaluator=lambda *a: {"db": None, "communicate": [False]}
                )
            output = await loop._run_trajectory({}, trajectory_id=f"id{i}", extra_info={"task_id": "0"})
            assert not loop.test_judge_calls
            record = next(loop.store.records())
            assert record.judge_result is None and record.metadata["simple_turn_labels"] is None
            assert record.custom_reward.train_reward == (1. if different and i == 7 else 1.5)
            if i == 0:
                from tau2_agentic_rl.scoring_retry import retry_scoring
                record.metadata["failure_phase"] = "reward_scoring"
                record.custom_reward = None
                assert await retry_scoring(record, None, loop.store)
                assert record.custom_reward.train_reward == 1.5 and record.judge_result is None
            return output.extra_fields
        extras = await asyncio.wait_for(asyncio.gather(*[
            group.session(i, member(i)) for i in range(8)
        ]), 20)
        cfg = CreditConfig(version="procredit-turn-v4", final_turn_rule="simple-turn-v2",
                           progress_credit="immediate_delta_v1", full_official_group_terminal_only=True)
        keys = [f"group1_{i}_0" for i in range(8)]
        report = queue_group_reports(keys, extras, cfg)["group1"]
        assert report["credit"]["has_signal"] is different
        for i, row in enumerate(report["credit"]["trajectories"]):
            advantage = row["trajectory_advantage"]
            if not different:
                assert advantage == 0
            else:
                assert (advantage < 0) if i == 7 else (advantage > 0)
            assert row["token_advantages"] == [
                advantage if turn >= 0 else 0 for turn in extras[i]["procredit"]["response_turn_ids"]
            ]
        with pytest.raises(ValueError):
            queue_group_reports([key.replace("group1", "other") for key in keys], extras, cfg)
    asyncio.run(run())
