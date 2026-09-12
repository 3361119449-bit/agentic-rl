"""Bounded scoring workers plus the real Judge's shared HTTP budget (no APIs)."""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
from test_review_boundaries import scoring_failure_record

from tau2_agentic_rl import concurrency, scoring_retry
from tau2_agentic_rl.judge import client as judge_client
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.schemas import JudgeResult
from tau2_agentic_rl.scoring_retry import retry_scoring_batch
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.versions import sha256_json


@pytest.mark.parametrize("count", [0, 1, 3, 4, 11])
def test_batch_has_at_most_four_workers_and_returns_every_result_in_order(
    monkeypatch, count
):
    async def check():
        ready, release = asyncio.Event(), asyncio.Event()
        started, active, peak = [], 0, 0

        async def retry(record, judge, store):
            nonlocal active, peak
            started.append(int(record.trajectory_id))
            active += 1
            peak = max(peak, active)
            if active == min(count, 4):
                ready.set()
            try:
                await release.wait()
                await asyncio.sleep(
                    0
                )  # Allow completion and worker refill to interleave.
                return int(record.trajectory_id) % 2 == 0
            finally:
                active -= 1

        monkeypatch.setattr(scoring_retry, "retry_scoring", retry)
        records = [SimpleNamespace(trajectory_id=str(i)) for i in range(count)]
        batch = asyncio.create_task(retry_scoring_batch(records, None, None))
        if count:
            await asyncio.wait_for(ready.wait(), timeout=5)
            assert len(started) == active == min(count, 4)
            release.set()
        assert await asyncio.wait_for(batch, timeout=5) == [
            i % 2 == 0 for i in range(count)
        ]
        assert sorted(started) == list(range(count))
        assert peak == min(count, 4) and active == 0

    asyncio.run(check())


def test_duplicate_ids_are_rejected_before_any_scoring(monkeypatch):
    async def unexpected(*args):
        pytest.fail("duplicate IDs must not reach the scorer or storage")

    monkeypatch.setattr(scoring_retry, "retry_scoring", unexpected)
    duplicate = SimpleNamespace(trajectory_id="same")
    with pytest.raises(ValueError, match="duplicate trajectory IDs"):
        asyncio.run(retry_scoring_batch([duplicate, duplicate], None, None))


def test_unexpected_worker_error_cancels_and_drains_siblings(monkeypatch):
    async def check():
        ready = asyncio.Event()
        active, started = 0, []

        async def retry(record, judge, store):
            nonlocal active
            active += 1
            started.append(record.trajectory_id)
            if active == 4:
                ready.set()
            try:
                await ready.wait()
                if record.trajectory_id == "0":
                    raise ValueError("invalid frozen scoring input")
                await asyncio.Event().wait()
            finally:
                active -= 1

        monkeypatch.setattr(scoring_retry, "retry_scoring", retry)
        records = [SimpleNamespace(trajectory_id=str(i)) for i in range(9)]
        with pytest.raises(ExceptionGroup) as error:
            await asyncio.wait_for(retry_scoring_batch(records, None, None), timeout=5)
        assert any(isinstance(exc, ValueError) for exc in error.value.exceptions)
        assert active == 0 and len(started) == 4

    asyncio.run(check())


def pending_records(count):
    records = []
    for index in range(count):
        record = scoring_failure_record()
        record.trajectory_id = f"record-{index}"
        # Distinct frozen inputs keep these tests from collapsing to cache hits.
        record.scoring_inputs["judge"]["task"] = {"id": index}
        record.metadata["scoring_inputs_sha256"] = sha256_json(record.scoring_inputs)
        records.append(record)
    return records


def configure_api_budget(monkeypatch, scratch_dir, api_limit):
    path = scratch_dir / "runtime_config.yaml"
    path.write_text(
        json.dumps(
            {
                "rollout": {
                    "max_active_trajectories": 1,
                    "user_api_max_inflight": 1,
                    "judge_api_max_inflight": api_limit,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTIC_RL_CONFIG", str(path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture")
    monkeypatch.setattr(concurrency, "_local_budgets", {})


def mock_transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        judge_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            **kwargs,
            transport=httpx.MockTransport(handler),
        ),
    )


@pytest.mark.parametrize("api_limit", [1, 2, 6])
@pytest.mark.parametrize("retry_http", [False, True])
def test_real_judge_http_is_limited_separately_from_four_scoring_tasks(
    monkeypatch, scratch_dir, api_limit, retry_http
):
    configure_api_budget(monkeypatch, scratch_dir, api_limit)
    active_api, peak_api, active_scoring, peak_scoring = 0, 0, 0, 0
    attempts = Counter()

    async def handle(request):
        nonlocal active_api, peak_api
        # Actual post() calls pass through DeepSeekJudge and api_budget().
        key = request.content
        attempts[key] += 1
        attempt = attempts[key]
        active_api += 1
        peak_api = max(peak_api, active_api)
        try:
            await asyncio.sleep(0.005)
            if retry_http and attempt == 1:
                return httpx.Response(429, json={"error": "retry fixture"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": JudgeResult().model_dump_json(),
                            }
                        }
                    ]
                },
            )
        finally:
            active_api -= 1

    mock_transport(monkeypatch, handle)

    async def quick_backoff(_):
        await asyncio.sleep(0.001)

    monkeypatch.setattr(judge_client, "asyncio", SimpleNamespace(sleep=quick_backoff))

    class Judge(DeepSeekJudge):
        async def evaluate(self, **inputs):
            nonlocal active_scoring, peak_scoring
            active_scoring += 1
            peak_scoring = max(peak_scoring, active_scoring)
            try:
                return await super().evaluate(**inputs)
            finally:
                active_scoring -= 1

    judge = Judge(
        JudgeConfig(
            model="fixture", cache_dir=str(scratch_dir / "cache"), max_retries=1
        )
    )
    records = pending_records(10)
    before = {r.trajectory_id: r.model_dump() for r in records}
    store = TrajectoryStore(scratch_dir / "records", attach_evaluation_identity=False)
    assert asyncio.run(retry_scoring_batch(records, judge, store)) == [True] * 10
    assert peak_scoring == 4
    assert peak_api == min(4, api_limit)
    assert active_scoring == active_api == 0
    budget = concurrency.api_budget().call("snapshot")
    assert budget["peak_judge_api_inflight"] == peak_api
    assert budget["active"]["judge_api"] == 0
    assert len(attempts) == 10 and set(attempts.values()) == (
        {2} if retry_http else {1}
    )
    saved = list(store.records())
    assert len(saved) == 10
    for record in saved:
        original = before[record.trajectory_id]
        for field in (
            "trajectory_id",
            "environment_seed",
            "official_scores",
            "scoring_inputs",
            "messages",
            "token_turns",
        ):
            assert record.model_dump()[field] == original[field]
        assert record.custom_reward is not None
        assert record.metadata["scoring_retries"][-1]["success"]


def test_batch_cancellation_releases_real_api_lease_and_keeps_records_pending(
    monkeypatch, scratch_dir
):
    configure_api_budget(monkeypatch, scratch_dir, 1)

    async def check():
        started = asyncio.Event()
        active_api = 0

        async def handle(request):
            nonlocal active_api
            active_api += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                active_api -= 1

        mock_transport(monkeypatch, handle)
        judge = DeepSeekJudge(
            JudgeConfig(model="fixture", cache_dir=str(scratch_dir / "cache"))
        )
        store = TrajectoryStore(
            scratch_dir / "records", attach_evaluation_identity=False
        )
        records = pending_records(8)
        for record in records:
            store.save(record)
        before = {p.name: p.read_bytes() for p in store.root.glob("*.json")}
        batch = asyncio.create_task(retry_scoring_batch(records, judge, store))
        await asyncio.wait_for(started.wait(), timeout=5)
        batch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(batch, timeout=5)
        assert active_api == 0
        assert concurrency.api_budget().call("snapshot")["active"]["judge_api"] == 0
        assert {p.name: p.read_bytes() for p in store.root.glob("*.json")} == before
        assert not list((scratch_dir / "cache").glob("*.json"))

    asyncio.run(check())
