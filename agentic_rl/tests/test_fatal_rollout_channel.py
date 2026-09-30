import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sampling_isolation import make_buffer

from tau2_agentic_rl.concurrency import BudgetState
from tau2_agentic_rl.slot_recovery import SlotRecoveryExhausted


class LocalBudget:
    def __init__(self):
        self.state = BudgetState({"trajectories": 1, "user_api": 1, "judge_api": 1})

    def call(self, method, *args):
        return getattr(self.state, method)(*args)


def test_fatal_latch_is_job_wide_and_preserves_first_failure():
    budget = LocalBudget()
    first = {
        "kind": "fatal",
        "uid": "bad",
        "phase": "token_alignment",
        "error_type": "ValueError",
    }
    budget.call("record_rollout_failure", first)
    budget.call(
        "record_rollout_failure", {"kind": "transient_exhausted", "uid": "other"}
    )
    budget.call("record_rollout_failure", {"kind": "fatal", "uid": "later"})
    assert budget.call("fatal_rollout") == first


def test_fatal_group_stops_before_returning_good_groups(scratch_dir):
    from tau2_agentic_rl.failures import FatalRolloutError

    budget = LocalBudget()
    buffer = make_buffer(scratch_dir)
    buffer.fatal_error_fn = lambda: budget.call("fatal_rollout")
    budget.call(
        "record_rollout_failure",
        {"kind": "fatal", "uid": "bad", "phase": "token_alignment"},
    )
    with pytest.raises(FatalRolloutError, match="token_alignment"):
        buffer.sample(1, "train", 4)
    assert not buffer.cleared


def test_transient_exhaustion_keeps_good_groups_trainable(scratch_dir):
    budget = LocalBudget()
    buffer = make_buffer(scratch_dir)
    buffer.fatal_error_fn = lambda: budget.call("fatal_rollout")
    budget.call("record_rollout_failure", {"kind": "transient_exhausted", "uid": "bad"})
    batch, _ = buffer.sample(1, "train", 4)
    assert len(batch) == 32 and all(not key.startswith("bad_") for key in batch)


def test_fatal_blocks_actor_update_after_batch_was_selected(scratch_dir):
    from test_procredit_runtime import production_method

    from tau2_agentic_rl.failures import FatalRolloutError

    calls = []

    class Parent:
        def _update_actor(self, batch, metrics):
            calls.append("optimizer")

    trainer = production_method("CappedPPOTrainerSync", "_update_actor", Parent)()
    trainer.replay_buffer = make_buffer(scratch_dir)
    trainer.replay_buffer.fatal_error_fn = lambda: {
        "uid": "bad",
        "phase": "token_alignment",
    }
    with pytest.raises(FatalRolloutError):
        trainer._update_actor("selected-batch", {})
    assert not calls


def test_fatal_checked_in_parent_validation_metadata_poll(scratch_dir):
    from tau2_agentic_rl.failures import FatalRolloutError

    buffer = make_buffer(scratch_dir)
    buffer.fatal_error_fn = lambda: {"uid": "bad-val", "phase": "token_alignment"}
    with pytest.raises(FatalRolloutError):
        buffer._sync_metadata_from_transfer_queue()


def test_runtime_warmup_rejected_before_training_or_checkpointing():
    from test_procredit_runtime import production_method

    trainer = production_method("CappedPPOTrainerSync", "fit", object)()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(critic_warmup=100))
    with pytest.raises(ValueError, match="critic_warmup=0"):
        trainer.fit(None)


def worker_class(budget, errors, queue, pending=None):
    path = Path(__file__).parents[1] / "src/tau2_agentic_rl/verl_failure_channel.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    # Replace only the Ray actor and TransferQueue transport, executing the
    # complete production worker (including session settlement).
    parsed.body = [
        n
        for n in parsed.body
        if not (
            isinstance(n, ast.Import)
            and any(a.name in {"ray", "transfer_queue"} for a in n.names)
            or isinstance(n, ast.ImportFrom)
            and (n.module or "").startswith("verl.")
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_TQWorker" for t in n.targets)
        )
    ]

    class Parent:
        def __init__(self):
            self.config = SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8))
            )
            self.started, self.ready = 0, asyncio.Event()

        async def _run_agent_loop(self, params, **kwargs):
            slot = kwargs["session_id"]
            if pending is not None:
                self.started += 1
                if self.started == 8:
                    self.ready.set()
                await self.ready.wait()
            if slot in errors:
                raise errors[slot]
            if pending is not None:
                await pending.wait()
            queue.append(("output", slot))

    async def put(**kwargs):
        queue.append(kwargs["tag"]["status"])

    scope = {
        "_TQWorker": Parent,
        "AgentLoopManagerTQ": object,
        "ray": SimpleNamespace(remote=lambda cls: cls),
        "tq": SimpleNamespace(async_kv_put=put),
        "api_budget": lambda: budget,
    }
    # api_budget is project code; inject the isolated budget after import.
    exec(compile(parsed, str(path), "exec"), scope)
    scope["api_budget"] = lambda: budget
    return scope["FailureAwareAgentLoopWorkerTQ"]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("config"),
        AssertionError("alignment"),
        AttributeError("bug"),
        RuntimeError("unknown"),
    ],
)
def test_worker_reports_fatal_before_pending_siblings_finish(error):
    async def scenario():
        budget, queue, finish = LocalBudget(), [], asyncio.Event()
        worker = worker_class(budget, {3: error}, queue, finish)()
        task = asyncio.create_task(
            worker._run_prompt({"uid": "bad"}, {}, {"validate": False})
        )
        try:
            for _ in range(20):
                await asyncio.sleep(0)
                if budget.call("fatal_rollout"):
                    break
            failure = budget.call("fatal_rollout")
            assert failure["uid"] == "bad" and failure["session_id"] == 3
            assert failure["error_type"] == type(error).__name__
            assert not task.done() and "failure" not in queue
        finally:
            finish.set()
            await task
        assert queue[-1] == "failure"
        assert len([item for item in queue if isinstance(item, tuple)]) == 7

    asyncio.run(scenario())


def test_worker_transient_exhaustion_drains_siblings_without_fatal_latch():
    budget, queue = LocalBudget(), []
    worker = worker_class(
        budget, {3: SlotRecoveryExhausted("bounded retries")}, queue
    )()
    asyncio.run(worker._run_prompt({"uid": "bad"}, {}, {"validate": False}))
    assert budget.call("fatal_rollout") is None
    assert queue[-1] == "failure"
    assert len([item for item in queue if isinstance(item, tuple)]) == 7


def test_worker_prompt_configuration_error_is_fatal():
    budget, queue = LocalBudget(), []
    worker = worker_class(budget, {}, queue)()
    asyncio.run(
        worker._run_prompt(
            {"uid": "bad", "__rollout_n__": "invalid"}, {}, {"validate": False}
        )
    )
    assert budget.call("fatal_rollout")["kind"] == "fatal"


def test_transient_sibling_does_not_hide_another_fatal_sibling():
    budget, queue = LocalBudget(), []
    worker = worker_class(
        budget,
        {1: SlotRecoveryExhausted("transient"), 3: ValueError("code bug")},
        queue,
    )()
    asyncio.run(worker._run_prompt({"uid": "bad"}, {}, {"validate": False}))
    assert budget.call("fatal_rollout")["session_id"] == 3


@pytest.mark.parametrize("status,retryable,attempts", [(401, False, 1), (503, True, 2)])
def test_real_judge_client_distinguishes_permanent_http_failure(
    monkeypatch, scratch_dir, status, retryable, attempts
):
    import httpx
    from test_judge_evidence import judge_inputs

    from tau2_agentic_rl.failures import JudgeServiceFailure, scoring_retryable
    from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig

    calls, original = [], httpx.AsyncClient

    def handle(request):
        calls.append(request)
        return httpx.Response(status, request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(**kw, transport=httpx.MockTransport(handle)),
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.delenv("AGENTIC_RL_CONFIG", raising=False)
    judge = DeepSeekJudge(
        JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1)
    )
    with pytest.raises(JudgeServiceFailure) as caught:
        asyncio.run(judge.evaluate(**judge_inputs()))
    assert scoring_retryable("judge", caught.value) is retryable
    assert len(calls) == attempts


@pytest.mark.parametrize(
    "error", [AttributeError("bug"), RuntimeError("unknown"), ValueError("bad config")]
)
def test_frozen_scoring_code_error_is_fatal_instead_of_exhaustion(scratch_dir, error):
    from test_review_boundaries import scoring_failure_record

    from tau2_agentic_rl.scoring_retry import retry_scoring
    from tau2_agentic_rl.storage import TrajectoryStore

    class Judge:
        async def evaluate(self, **kwargs):
            raise error

    record = scoring_failure_record()
    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    with pytest.raises(type(error)):
        asyncio.run(retry_scoring(record, Judge(), store))
    saved = next(store.records())
    assert saved.metadata["scoring_retries"][-1]["success"] is False


def test_frozen_scoring_timeout_remains_retryable(scratch_dir):
    from test_review_boundaries import scoring_failure_record

    from tau2_agentic_rl.scoring_retry import retry_scoring
    from tau2_agentic_rl.storage import TrajectoryStore

    class Judge:
        async def evaluate(self, **kwargs):
            raise TimeoutError("service")

    record = scoring_failure_record()
    assert not asyncio.run(
        retry_scoring(
            record,
            Judge(),
            TrajectoryStore(scratch_dir, attach_evaluation_identity=False),
        )
    )
