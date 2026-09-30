"""Exercise the production buffer with only the external queue transport mocked."""

import ast
import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.concurrency import BudgetState
from tau2_agentic_rl.failures import raise_if_fatal
from tau2_agentic_rl.procredit_runtime import _save_group_audit
from tau2_agentic_rl.slot_recovery import (
    RolloutInfrastructureError,
    SlotRecoveryExhausted,
    interaction_retryable,
    run_training_slot,
)


def buffer_class():
    path = Path(__file__).parents[1] / "src/tau2_agentic_rl/verl_capped_trainer.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    cap = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "DynamicSamplingCapReached")
    node = next(n for n in parsed.body
                if isinstance(n, ast.ClassDef) and n.name == "CappedDynamicReplayBuffer")
    for method in node.body:
        if isinstance(method, ast.FunctionDef):
            method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cap, node], type_ignores=[])
    ast.fix_missing_locations(module)

    class Parent:
        def __init__(self, **kwargs):
            self.refill_fn = lambda n: pytest.fail("a failed group must not trigger a refill")

        def _sync_metadata_from_transfer_queue(self):
            pass

        def _terminal_eviction_reasons(self, step, partition):
            return set(), set(), set(self.failure_keys[partition]), Counter()

        def _sampleable_terminal_keys(self, partition, reasons):
            return (self.finished_keys[partition] | self.failure_keys[partition]) - set.union(*reasons[:3])

        def _evict_terminal_groups(self, step, partition, reasons):
            uids = set.union(*reasons[:3])
            self._clear_groups(partition, uids)
            return uids, 0, 0, {}

        def _clear_groups(self, partition, uids):
            self.cleared.append(set(uids))
            self.partitions[partition] = {k: v for k, v in self.partitions[partition].items() if k.split("_")[0] not in uids}
            for status in (self.finished_keys, self.failure_keys):
                status[partition].difference_update(uids)

        def _select_prompt_uids(self, partition, uids, size):
            return sorted(uids)[:size], dict(self.partitions[partition]), {}

        def _materialize_batch(self, partition, uids, snapshot):
            return [k for k in snapshot if k.split("_")[0] in uids]

    scope = {
        "ReplayBuffer": Parent, "time": time, "json": json, "Counter": Counter, "dataclass": dataclass,
        "SlotRecoveryExhausted": SlotRecoveryExhausted,
        "DAPO_FILTERED_REWARD_COUNTS_KEY": "_dapo_filtered_reward_counts",
        "_accumulate_eviction_metrics": lambda *args: None,
        "_save_group_audit": _save_group_audit,
        "raise_if_fatal": raise_if_fatal,
    }
    exec(compile(module, str(path), "exec"), scope)
    return scope["CappedDynamicReplayBuffer"]


def make_buffer(scratch_dir):
    cls = buffer_class()
    buffer = cls(conceptual_gen_batch_size=8, max_num_gen_batches=3, rollout_group_size=8,
                 credit_config=CreditConfig(version="procredit-turn-v4"), group_audit_dir=scratch_dir)
    buffer.finished_keys = {"train": {f"good{i}" for i in range(7)}}
    buffer.failure_keys = {"train": {"bad"}}
    buffer.pending_keys, buffer.running_keys = {"train": set()}, {"train": set()}
    buffer.partitions = {"train": {f"good{g}_{s}_0": {"status": "finished"} for g in range(7) for s in range(8)}}
    buffer.partitions["train"].update({f"bad_{s}_0": {"status": "finished"} for s in range(7)})
    buffer.prompt_global_steps = {"train": dict.fromkeys(buffer.finished_keys["train"] | {"bad"}, 1)}
    buffer.cleared = []
    return buffer


def test_one_failed_group_does_not_stop_good_groups_or_train_seven_siblings(scratch_dir):
    buffer = make_buffer(scratch_dir)
    batch, metrics = buffer.sample(1, "train", 4)
    assert len(batch) == 32
    assert {k.split("_")[0] for k in batch} == {f"good{i}" for i in range(4)}
    assert not any(k.startswith("bad_") for k in batch)
    assert metrics["training/dynamic_sampling/quarantined_groups"] == 1
    audit = json.loads(next((scratch_dir / "quarantined").glob("*.json")).read_text())
    assert audit["materializable_members"] == [f"bad_{s}_0" for s in range(7)]
    assert "bad" not in buffer.failure_keys["train"]  # Cleanup only after draining.


def test_failed_siblings_survive_until_other_groups_finish(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer.running_keys["train"] = {"good6"}
    buffer.finished_keys["train"].remove("good6")
    def poll(partition, stamp):
        assert all(f"bad_{s}_0" in buffer.partitions[partition] for s in range(7))
        buffer.running_keys[partition].clear()
        buffer.finished_keys[partition].add("good6")
        return stamp
    buffer._wait_for_next_poll = poll
    batch, metrics = buffer.sample(1, "train", 4)
    assert len(batch) == 32 and metrics["training/dynamic_sampling/quarantined_groups"] == 1


def test_failed_groups_do_not_bypass_the_logical_generation_cap(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer.finished_keys["train"] = {"good0", "good1"}
    buffer.failure_keys["train"] = {"bad", *(f"good{i}" for i in range(2, 7))}
    refills = []
    buffer.refill_fn = refills.append
    with pytest.raises(RuntimeError) as error:
        buffer.sample(1, "train", 4)
    assert error.value.generated_trajectories == 192
    assert error.value.metrics["training/dynamic_sampling/quarantined_groups"] == 6
    assert refills == [8, 8]
    assert not buffer.partitions["train"]
    assert len(list((scratch_dir / "quarantined").glob("*.json"))) == 6


def test_quarantine_audit_is_idempotent_if_sampling_is_reentered(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer._audit_quarantined_groups("train", {"bad"})
    buffer._audit_quarantined_groups("train", {"bad"})
    assert len(list((scratch_dir / "quarantined").glob("*.json"))) == 1


def test_trainer_snapshots_counts_before_first_batch_is_submitted(scratch_dir):
    from test_procredit_runtime import production_method
    state = BudgetState({"trajectories": 1, "user_api": 1, "judge_api": 1})
    buffer = make_buffer(scratch_dir)
    buffer.rollout_counts_fn = lambda: state.rollout_counts("train")
    for _ in range(5):
        state.record_rollout_attempt("train", "initial")
    cls = production_method("CappedPPOTrainerSync", "step", object)
    trainer = cls()
    from types import SimpleNamespace
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(critic_warmup=0))
    trainer.parameter_sync_step, trainer.conceptual_gen_batch_size = 1, 8
    trainer.replay_buffer = buffer
    def submit(n):
        for _ in range(n * 8):
            state.record_rollout_attempt("train", "initial")
    def once(metrics, timing, **kwargs):
        buffer._add_sampling_metrics(metrics, 1, 7)
        return "batch"
    trainer._add_prompts_to_generate, trainer._step_once = submit, once
    metrics = {}
    assert trainer.step(metrics, {}) == "batch"
    assert metrics["training/dynamic_sampling/physical_rollout_attempts"] == 64


def test_exception_cycles_do_not_count_as_transient_evidence():
    error = RuntimeError("cyclic wrapper")
    error.__cause__ = error
    assert not interaction_retryable("environment_reset", error)


def test_asyncio_timeout_is_recoverable_but_outer_cancellation_is_not():
    async def scenario():
        with pytest.raises(TimeoutError) as caught:
            await asyncio.wait_for(asyncio.sleep(10), timeout=0.001)
        assert isinstance(caught.value.__cause__, asyncio.CancelledError)
        assert interaction_retryable("model_generation", caught.value)
    asyncio.run(scenario())
    assert not interaction_retryable("model_generation", asyncio.CancelledError())


def test_real_httpx_anyio_timeout_chain_remains_recoverable():
    import anyio
    from httpcore._backends.anyio import AnyIOStream
    from httpx._transports.default import map_httpcore_exceptions
    class Stream:
        async def receive(self, **kwargs):
            await anyio.sleep(10)
    async def scenario():
        with pytest.raises(httpx.ReadTimeout) as caught:
            with map_httpcore_exceptions():
                await AnyIOStream(Stream()).read(1, timeout=0.001)
        assert interaction_retryable("model_generation", caught.value)
    asyncio.run(scenario())


def test_real_httpx_broken_stream_chain_requires_transport_context():
    import anyio
    from httpcore._backends.anyio import AnyIOStream
    from httpx._transports.default import map_httpcore_exceptions
    class Stream:
        async def receive(self, **kwargs):
            raise anyio.BrokenResourceError
    async def scenario():
        with pytest.raises(httpx.ReadError) as caught:
            with map_httpcore_exceptions():
                await AnyIOStream(Stream()).read(1, timeout=1)
        assert interaction_retryable("model_generation", caught.value)
    asyncio.run(scenario())
    assert not interaction_retryable("model_generation", anyio.BrokenResourceError())


@pytest.mark.parametrize("error", [RuntimeError("unknown"), AttributeError("bad field"),
                                  FileNotFoundError("missing model"), NotImplementedError("unsupported")])
def test_unknown_service_errors_never_replace_a_slot(error):
    assert not interaction_retryable("model_generation", error)
    calls = []
    async def generate(params, **kwargs):
        calls.append(kwargs)
        raise RolloutInfrastructureError("model_generation", kwargs["trajectory_id"]) from error
    with pytest.raises(RolloutInfrastructureError):
        asyncio.run(run_training_slot(generate, {}, project={"slot_recovery": {"max_resamples": 2}}, kwargs={}))
    assert len(calls) == 1


@pytest.mark.parametrize("status, expected", [(400, False), (401, False), (404, False), (408, True), (429, True), (500, True), (501, False), (503, True)])
def test_only_transient_http_statuses_allow_replacement(status, expected):
    request = httpx.Request("POST", "https://fixture.test")
    error = httpx.HTTPStatusError("fixture", request=request, response=httpx.Response(status, request=request))
    assert interaction_retryable("model_generation", error) is expected


def test_no_exception_evidence_is_not_a_transient_failure():
    assert not interaction_retryable("environment_reset")
    assert not interaction_retryable("model_generation", RolloutInfrastructureError("model_generation", "id"))


def test_v4_config_declares_group_quarantine_instead_of_stopping_all_sampling():
    from tau2_agentic_rl.config import load_yaml
    cfg = load_yaml(Path(__file__).parents[1] / "configs/rl/airline_procredit_v4.yaml")
    assert cfg["slot_recovery"]["on_exhaustion"] == "quarantine_group"


def test_physical_counts_include_failed_and_rejected_attempts_before_filtering(scratch_dir):
    state = BudgetState({"trajectories": 1, "user_api": 1, "judge_api": 1})
    for _ in range(5):  # Costs from a prior optimizer attempt.
        state.record_rollout_attempt("train", "initial")
    initial = state.rollout_counts("train")
    for _ in range(63):  # Other slots, including filtered groups.
        state.record_rollout_attempt("train", "initial")
    calls = []
    from tau2_agentic_rl.user_simulation import UserSimulationRejected
    async def observer(kind):
        state.record_rollout_attempt("train", kind)
    async def generate(params, **kwargs):
        calls.append(kwargs["trajectory_id"])
        if len(calls) == 1:
            raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"]) from ConnectionError("offline")
        raise UserSimulationRejected("invalid user")
    with pytest.raises(SlotRecoveryExhausted):
        asyncio.run(run_training_slot(generate, {}, project={
            "slot_recovery": {"max_resamples": 2},
            "user_sim_filter": {"enabled": True, "max_resamples": 1},
        }, kwargs={}, on_attempt=observer))
    assert state.rollout_counts("train") == {"initial": 69, "infrastructure": 1, "user": 1}
    buffer = make_buffer(scratch_dir)
    buffer.rollout_counts_fn = lambda: state.rollout_counts("train")
    buffer.rollout_counts_start = initial
    metrics = {}
    buffer._add_sampling_metrics(metrics, 1, 7)
    assert metrics["training/dynamic_sampling/generated_logical_rollouts"] == 64
    assert metrics["training/dynamic_sampling/physical_rollout_attempts"] == 66
    assert metrics["training/dynamic_sampling/infrastructure_replacements"] == 1
    assert metrics["training/dynamic_sampling/user_replacements"] == 1
    assert state.rollout_counts("validation") == {"initial": 0, "infrastructure": 0, "user": 0}
