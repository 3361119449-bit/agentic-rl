"""Exercise the production buffer with only the external queue transport mocked."""

import ast
import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.concurrency import BudgetState
from tau2_agentic_rl.failures import raise_if_fatal
from tau2_agentic_rl.procredit_runtime import (
    _save_group_audit,
    queue_group_reports,
    terminal_group_contract,
)
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

    class Batch(list):
        def __init__(self, keys, tags):
            super().__init__(keys)
            self.keys, self.tags, self.extra_info = keys, tags, {}

    class Parent:
        def __init__(self, **kwargs):
            self.refill_fn = lambda n: pytest.fail("a failed group must not trigger a refill")

        def _sync_metadata_from_transfer_queue(self):
            pass

        def _terminal_eviction_reasons(self, step, partition):
            constant, counts = self._dapo_filtered_keys(partition)
            materialized = {key.split("_")[0] for key in self.partitions[partition]}
            return set(), constant, self.failure_keys[partition] - materialized, counts

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
            keys = [k for k in snapshot if k.split("_")[0] in uids]
            return Batch(keys, [snapshot[k] for k in keys])

    transport = SimpleNamespace(extras={})
    transport.kv_batch_get = lambda **kw: {"extra_fields": [transport.extras[k] for k in kw["keys"]]}
    transport.kv_list = lambda: {}
    transport.kv_clear = lambda **kw: None

    scope = {
        "ReplayBuffer": Parent, "time": time, "json": json, "Counter": Counter, "dataclass": dataclass,
        "SlotRecoveryExhausted": SlotRecoveryExhausted,
        "DAPO_FILTERED_REWARD_COUNTS_KEY": "_dapo_filtered_reward_counts",
        "_accumulate_eviction_metrics": lambda *args: None,
        "_save_group_audit": _save_group_audit,
        "raise_if_fatal": raise_if_fatal,
        "queue_group_reports": queue_group_reports,
        "terminal_group_contract": terminal_group_contract,
        "tq": transport,
    }
    exec(compile(module, str(path), "exec"), scope)
    cls = scope["CappedDynamicReplayBuffer"]
    cls.transport = transport
    return cls


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
    from test_turn_local_v4 import rows
    for uid in [*buffer.finished_keys["train"], "bad"]:
        for session, row in enumerate(rows()):
            buffer.transport.extras[f"{uid}_{session}_0"] = {"procredit": row}
    buffer.terminal_groups = {"train": {
        uid: {"is_prompt": True, "status": "failure" if uid == "bad" else "finished",
              "rollout_n": 8, "failed_session_ids": [7] if uid == "bad" else [],
              "failure_kind": "transient_exhausted" if uid == "bad" else None}
        for uid in buffer.prompt_global_steps["train"]
    }}
    terminal_metadata = buffer.terminal_groups
    buffer.terminal_metadata_fn = lambda: terminal_metadata
    return buffer


def test_one_failed_slot_keeps_seven_siblings_and_all_seven_good_groups(scratch_dir):
    buffer = make_buffer(scratch_dir)
    batch, metrics = buffer.sample(1, "train", 4)
    assert len(batch) == 63
    assert {k.split("_")[0] for k in batch} == {"bad", *(f"good{i}" for i in range(7))}
    assert metrics["training/dynamic_sampling/selected_real_rollouts"] == 63
    assert metrics["training/dynamic_sampling/salvaged_rollouts"] == 7
    assert metrics["training/dynamic_sampling/selected_groups"] == 8
    assert batch.extra_info["procredit_terminal_groups"]["bad"]["failed_session_ids"] == [7]
    audit = json.loads(next(p for p in scratch_dir.glob("*.json") if json.loads(p.read_text())["uid"] == "bad").read_text())
    assert audit["members"] == [f"bad_{s}_0" for s in range(7)]
    assert not any(buffer.cleared)  # No good surplus or partial group is discarded.


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
    assert len(batch) == 63 and metrics["training/dynamic_sampling/salvaged_rollouts"] == 7


def test_failed_groups_do_not_bypass_the_logical_generation_cap(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer.finished_keys["train"] = {"good0", "good1"}
    buffer.failure_keys["train"] = {"bad", *(f"good{i}" for i in range(2, 7))}
    for uid in buffer.failure_keys["train"]:
        buffer.terminal_groups["train"][uid].update(status="failure", failed_session_ids=list(range(8)), failure_kind="transient_exhausted")
    buffer.partitions["train"] = {k: v for k, v in buffer.partitions["train"].items() if k.split("_")[0] in buffer.finished_keys["train"]}
    refills = []
    buffer.refill_fn = refills.append
    batch, metrics = buffer.sample(1, "train", 4)
    assert len(batch) == 16
    assert metrics["training/dynamic_sampling/generated_logical_rollouts"] == 192
    assert metrics["training/dynamic_sampling/cap_remainder_used"] == 1
    assert metrics["training/dynamic_sampling/selected_groups"] == 2
    assert refills == [8, 8]
    assert all(k.split("_")[0] in {"good0", "good1"} for k in batch)


def test_cap_skips_only_when_no_group_has_real_signal(scratch_dir):
    buffer = make_buffer(scratch_dir)
    for extra in buffer.transport.extras.values():
        extra["procredit"].update(valid=True)
        extra["procredit"]["policy_credit"]["violating_turns"] = []
    refills = []
    buffer.refill_fn = refills.append
    with pytest.raises(RuntimeError) as error:
        buffer.sample(1, "train", 4)
    assert error.value.generated_trajectories == 192
    assert refills == [8, 8]
    assert not buffer.partitions["train"]


def test_failed_slot_output_written_before_error_is_never_used(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer.partitions["train"]["bad_7_0"] = {"status": "finished"}
    batch, _ = buffer.sample(1, "train", 4)
    assert len(batch) == 63
    assert "bad_7_0" not in batch


def test_missing_successful_slot_stops_instead_of_refilling(scratch_dir):
    buffer = make_buffer(scratch_dir)
    del buffer.partitions["train"]["bad_6_0"]
    with pytest.raises(ValueError, match="incomplete"):
        buffer.sample(1, "train", 4)


def test_even_a_full_group_with_stale_policy_is_not_used(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer.prompt_global_steps["train"]["good0"] = 0
    with pytest.raises(ValueError, match="policy"):
        buffer.sample(1, "train", 4)


def test_refill_consumes_prior_good_groups_and_all_new_good_groups(scratch_dir):
    from test_turn_local_v4 import rows

    buffer = make_buffer(scratch_dir)
    # The first wave leaves two viable groups; all other final advantages are zero.
    for key, extra in buffer.transport.extras.items():
        if key.split("_")[0] not in {"good0", "good1"}:
            extra["procredit"]["valid"] = True
            extra["procredit"]["policy_credit"]["violating_turns"] = []
    refills = []

    def submit(n):
        refills.append(n)
        for g in range(n):
            uid = f"next{g}"
            buffer.finished_keys["train"].add(uid)
            buffer.prompt_global_steps["train"][uid] = 1
            buffer.terminal_metadata_fn()["train"][uid] = {
                "status": "finished", "rollout_n": 8,
                "failed_session_ids": [], "failure_kind": None,
            }
            for s, row in enumerate(rows()):
                key = f"{uid}_{s}_0"
                buffer.transport.extras[key] = {"procredit": row}
                buffer.partitions["train"][key] = {"status": "finished"}

    buffer.refill_fn = submit
    batch, metrics = buffer.sample(1, "train", 4)
    assert refills == [8]
    assert len(batch) == 80
    assert {k.split("_")[0] for k in batch} == {"good0", "good1", *(f"next{i}" for i in range(8))}
    assert metrics["training/dynamic_sampling/selected_groups"] == 10
    assert metrics["training/dynamic_sampling/generated_logical_rollouts"] == 128
    assert metrics["training/dynamic_sampling/selected_real_rollouts"] == 80
    assert metrics["training/dynamic_sampling/rollout_use_rate"] == .625


def test_exhaustion_audit_is_idempotent_if_sampling_is_reentered(scratch_dir):
    buffer = make_buffer(scratch_dir)
    buffer._audit_exhausted_groups("train", {"bad"})
    buffer._audit_exhausted_groups("train", {"bad"})
    assert len(list((scratch_dir / "exhausted").glob("*.json"))) == 1


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


def test_v4_config_declares_group_salvage():
    from tau2_agentic_rl.config import load_yaml
    cfg = load_yaml(Path(__file__).parents[1] / "configs/rl/airline_procredit_v4.yaml")
    assert cfg["slot_recovery"]["on_exhaustion"] == "salvage_group"


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
