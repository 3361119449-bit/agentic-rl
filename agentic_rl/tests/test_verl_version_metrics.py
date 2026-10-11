"""Execute pinned TQ/metrics methods with CPU transport/tensor stand-ins.

Set VERL_VERSION_CONTRACT_SOURCE to a pinned veRL checkout, or a directory
containing its agent_loop_tq.py and trainer_base.py. No GPU/PPO claim is made:
the complete upstream methods run, but unrelated tensor metrics are stubbed.
"""

import ast
import asyncio
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_procredit_rollout import rollout_fixture

np = pytest.importorskip("numpy")


def upstream_method(filename, owner, name, scope):
    root = os.environ.get("VERL_VERSION_CONTRACT_SOURCE")
    if root is None:
        pytest.skip("requires pinned veRL source via VERL_VERSION_CONTRACT_SOURCE")
    path = Path(root) / "verl/trainer/ppo/v1" / filename
    if not path.exists():
        path = Path(root) / filename
    # Pin the entire source file, not a reimplementation of its metadata logic.
    expected = {
        "agent_loop_tq.py": "43b7a2fca55e46c2fe0d7844257aa57c792971dd557a8c2b03b6cfef05a2dc80",
        "trainer_base.py": "74b8df434e4046933a4c7b7a949f3540e39eb07ac44ceb7033dedc8b27abe5c4",
    }
    source = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == expected[filename]
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in cls.body if getattr(n, "name", None) == name)
    module = ast.Module(body=[method], type_ignores=[])
    scope = {"__name__": "pinned_verl_contract", **scope}
    # Postponed annotations avoid importing GPU-only type dependencies.
    import __future__
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


class Tensor:
    """Only tensor plumbing needed by the two unmodified upstream methods."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def size(self, dim):
        return self.values.shape[dim]

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.values, dim))

    def squeeze(self, dim):
        return Tensor(np.squeeze(self.values, dim))

    def offsets(self):
        return Tensor([0, self.values.size])

    def diff(self):
        return Tensor(np.diff(self.values))

    def __add__(self, other):
        return Tensor(self.values + other.values)

    def tolist(self):
        return self.values.tolist()

    def float(self):
        return self


class MetricData(dict):
    def to_padded_tensor(self):
        return self


@pytest.mark.parametrize("server_bounds", [True, False])
def test_first_and_second_step_through_real_tq_and_metrics(scratch_dir, server_bounds):
    queued = []

    async def put(**kwargs):
        queued.append(kwargs)

    async def no_op(*args, **kwargs):
        pass

    worker_method = upstream_method("agent_loop_tq.py", "AgentLoopWorkerTQ", "_agent_loop_postprocess", {
        "torch": SimpleNamespace(
            tensor=lambda x, **kw: Tensor(x), int64="int64",
            cat=lambda xs, **kw: Tensor(np.concatenate([x.values for x in xs])),
            ones_like=lambda x, **kw: Tensor(np.ones_like(x.values)),
        ),
        "tq": SimpleNamespace(async_kv_batch_put=put),
        "list_of_dict_to_tensordict": lambda fields: fields,
    })
    worker = SimpleNamespace(
        _compute_score=no_op, _compute_teacher_logprobs=no_op,
        _compute_multi_modal_inputs=lambda *a: {},
        _compute_position_ids=lambda ids, *a: ids,
    )
    metric_scope = {
        "np": np, "tq": SimpleNamespace(kv_batch_get=lambda **kw: None),
        "get_metric_data_with_optional_routed_experts": lambda **kw: MetricData(
            prompts=Tensor([1, 2]), responses=Tensor([4, 9, 7, 5, 9, 7, 6]),
            num_turns=Tensor([3]), rm_scores=Tensor([0.0]),
        ),
        "DataProto": lambda **kw: SimpleNamespace(select_idxs=lambda mask: kw),
    }
    for name in ("compute_moe_lb_metrics", "compute_data_metrics", "compute_timing_metrics",
                 "compute_throughout_metrics", "compute_variance_proxy_metrics", "compute_spec_decode_metrics"):
        metric_scope[name] = lambda *a, **kw: {}
    metrics_method = upstream_method("trainer_base.py", "PPOTrainer", "_compute_metrics", metric_scope)
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout={}, model=SimpleNamespace())),
        _rollout_moe_lb_metrics_accumulator=None, use_critic=False,
        _get_n_gpus_for_throughput=lambda: 1,
    )
    for version in (0, 1):
        fields = ({"min_global_steps": version, "max_global_steps": version}
                  if server_bounds else {"global_steps": version})
        _, output, _ = rollout_fixture(scratch_dir / str(version), version_fields=[fields] * 3)
        output.as_dict = lambda: {
            **vars(output), "prompts": Tensor(output.prompt_ids),
            "responses": Tensor(output.response_ids),
        }
        asyncio.run(worker_method(worker, output, False, uid="prompt", session_id=str(version),
                                  global_steps=version + 1))
        packet = queued[-1]
        batch = SimpleNamespace(tags=packet["tags"], keys=packet["keys"], partition_id="train")
        metrics = {}
        metrics_method(trainer, batch, metrics, {}, global_steps=version + 1, epoch=0)
        assert metrics["training/off_policy/trajectory_spans/mean"] == 1
        assert metrics["training/off_policy/trajectory_staleness/mean"] == 0
        assert metrics["training/off_policy/trajectory_staleness_worst/mean"] == 0
        assert packet["tags"][0]["min_global_steps"] == version
        assert packet["tags"][0]["max_global_steps"] == version
