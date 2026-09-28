import ast
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_policy_credit import policy_row, v2
from test_procredit_credit import row

from tau2_agentic_rl import procredit_runtime as runtime
from tau2_agentic_rl.advantages import CreditConfig


def queue_rows(uid="group", offset=0, **changes):
    keys = [f"{uid}_{i}_0" for i in range(8)]
    extras = [{"procredit": row(i + offset, **changes)} for i in range(8)]
    return keys, extras


def production_method(class_name, method, parent, **scope):
    path = Path(__file__).parents[1] / "src/tau2_agentic_rl/verl_capped_trainer.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    function = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )
    module = ast.parse(
        "from __future__ import annotations\nclass UnderTest(Parent):\n    pass\n"
    )
    module.body[1].body = [function]
    ast.fix_missing_locations(module)
    namespace = {
        "Parent": parent,
        "Counter": Counter,
        "queue_group_reports": runtime.queue_group_reports,
        "build_credit_tensors": runtime.build_credit_tensors,
        **scope,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["UnderTest"]


def test_queue_uses_real_uid_and_preserves_original_membership(scratch_dir):
    ka, ea = queue_rows("a")
    kb, eb = queue_rows("b", offset=8, terminal_success=0, score=0.5)
    result = runtime.queue_group_reports(
        ka + kb, ea + eb, CreditConfig(), audit_dir=scratch_dir
    )
    assert set(result) == {"a", "b"}
    assert result["a"]["credit"]["turn_mean"] == 1.25
    assert result["b"]["credit"]["turn_mean"] == 0.25
    saved = [json.loads(p.read_text()) for p in scratch_dir.glob("*.json")]
    assert {item["uid"] for item in saved} == {"a", "b"}
    assert sorted(len(item["members"]) for item in saved) == [8, 8]
    runtime.queue_group_reports(ka, ea, CreditConfig(), audit_dir=scratch_dir)
    ea[0]["procredit"]["score"] = 1.4
    with pytest.raises(ValueError, match="audit"):
        runtime.queue_group_reports(ka, ea, CreditConfig(), audit_dir=scratch_dir)


def test_missing_or_duplicate_session_never_pads_a_group():
    keys, extras = queue_rows()
    with pytest.raises(ValueError, match="complete"):
        runtime.queue_group_reports(keys[:-1], extras[:-1], CreditConfig())
    keys[-1] = "group_0_1"
    with pytest.raises(ValueError, match="session"):
        runtime.queue_group_reports(keys, extras, CreditConfig())


def test_actual_buffer_does_not_call_scalar_filter_for_new_mode():
    keys, extras = queue_rows()

    class Parent:
        def _dapo_filtered_keys(self, partition):
            raise AssertionError("scalar filtering must not run")

    cls = production_method(
        "CappedDynamicReplayBuffer",
        "_dapo_filtered_keys",
        Parent,
        tq=SimpleNamespace(kv_batch_get=lambda **kwargs: {"extra_fields": extras}),
    )
    buffer = cls()
    buffer.credit_config, buffer.group_audit_dir = CreditConfig(), None
    buffer.credit_cache = {}
    buffer.finished_keys = {"train": {"group"}}
    buffer.partitions = {"train": dict.fromkeys(keys)}
    assert buffer._dapo_filtered_keys("train")[0] == set()
    buffer.credit_cache = {}
    for extra in extras:
        extra["procredit"].update(valid=False, score=0, terminal_success=0)
    assert buffer._dapo_filtered_keys("train")[0] == {"group"}


@pytest.mark.parametrize("policy_local", [False, True])
def test_real_torch_nested_advantages_reach_production_trainer(policy_local):
    torch = pytest.importorskip("torch")
    keys, extras = queue_rows()
    config = CreditConfig()
    expected = [0.25, 0.25, 0, -0.25, -0.25]
    if policy_local:
        extras = [{"procredit": policy_row(i)} for i in range(8)]
        config = v2()
        expected = [0.25, 0.25, 0, 0, -1.25]
    mask = torch.nested.as_nested_tensor(
        [torch.tensor([1, 1, 0, 1, 1]) for _ in keys], layout=torch.jagged
    )
    stored = {"response_mask": mask, "extra_fields": extras}
    old = torch.tensor([-1.0])
    stored["old_log_probs"] = old

    def put(**kwargs):
        stored.update(kwargs["fields"])
        return batch

    class Parent:
        def _compute_advantage(self, *args, **kwargs):
            raise AssertionError("scalar GRPO path must not overwrite turn credit")

    cls = production_method(
        "CappedPPOTrainerSync",
        "_compute_advantage",
        Parent,
        tq=SimpleNamespace(kv_batch_get=lambda **kw: stored, kv_batch_put=put),
        TensorDict=lambda fields, **kwargs: fields,
    )
    trainer = cls()
    trainer._procredit_runtime = lambda: (config, None)
    batch = SimpleNamespace(keys=keys, partition_id="train")
    metrics = {}
    assert trainer._compute_advantage(batch, metrics) is batch
    assert stored["advantages"].unbind()[0].tolist() == expected
    assert stored["returns"].unbind()[0].tolist() == expected
    assert stored["old_log_probs"] is old
    assert metrics["procredit/nonzero_token_fraction"] == (0.75 if policy_local else 1)
    if policy_local:
        assert metrics["procredit/policy_violation_turns"] == 8
        assert metrics["procredit/constant_score_with_turn_signal"] == 1


def test_zero_score_policy_group_survives_production_buffer_filter():
    keys, _ = queue_rows()
    extras = [{"procredit": policy_row(i, phi=[0, 0, 0, 0])} for i in range(8)]

    class Parent:
        def _dapo_filtered_keys(self, partition):
            raise AssertionError("scalar filtering must not run")

    cls = production_method(
        "CappedDynamicReplayBuffer", "_dapo_filtered_keys", Parent,
        tq=SimpleNamespace(kv_batch_get=lambda **kwargs: {"extra_fields": extras}),
    )
    buffer = cls()
    buffer.credit_config, buffer.group_audit_dir = v2(), None
    buffer.credit_cache = {}
    buffer.finished_keys = {"train": {"group"}}
    buffer.partitions = {"train": dict.fromkeys(keys)}
    assert buffer._dapo_filtered_keys("train")[0] == set()


def test_mask_mismatch_is_rejected_instead_of_training_observations():
    torch = pytest.importorskip("torch")
    keys, extras = queue_rows()
    mask = torch.nested.as_nested_tensor(
        [torch.tensor([1, 1, 1, 1, 1]) for _ in keys], layout=torch.jagged
    )
    with pytest.raises(ValueError, match="mask"):
        runtime.build_credit_tensors(keys, extras, mask, CreditConfig())


def test_v3_zero_score_process_group_reaches_buffer_and_real_torch():
    torch = pytest.importorskip("torch")
    from test_process_credit import process_rows

    keys, _ = queue_rows()
    extras = [{"procredit": row} for row in process_rows()]
    config = CreditConfig(version="procredit-turn-v3")

    class Parent:
        def _dapo_filtered_keys(self, partition):
            raise AssertionError("scalar filtering must not run")

    cls = production_method(
        "CappedDynamicReplayBuffer", "_dapo_filtered_keys", Parent,
        tq=SimpleNamespace(kv_batch_get=lambda **kwargs: {"extra_fields": extras}),
    )
    buffer = cls()
    buffer.credit_config, buffer.group_audit_dir = config, None
    buffer.credit_cache = {}
    buffer.finished_keys = {"train": {"group"}}
    buffer.partitions = {"train": dict.fromkeys(keys)}
    assert buffer._dapo_filtered_keys("train")[0] == set()
    mask = torch.nested.as_nested_tensor(
        [torch.tensor([1, 1, 0, 1, 1]) for _ in keys], layout=torch.jagged
    )
    tensors, metrics = runtime.build_credit_tensors(keys, extras, mask, config)
    assert tensors["advantages"].unbind()[0].tolist() == pytest.approx([0, 0, 0, -.1, 0])
    assert metrics["procredit/process_cost_turns"] == 8


def test_real_tensordict_metadata_and_padded_advantages():
    torch = pytest.importorskip("torch")
    td = pytest.importorskip("tensordict")
    keys, extras = queue_rows()
    data = td.TensorDict(
        {
            "response_mask": torch.tensor([[1, 1, 0, 1, 1, 0, 0]] * 8),
            "extra_fields": td.NonTensorStack(*[td.NonTensorData(e) for e in extras]),
        },
        batch_size=8,
    )
    fields, _ = runtime.build_credit_tensors(
        keys, list(data["extra_fields"]), data["response_mask"], CreditConfig()
    )
    output = td.TensorDict(fields, batch_size=8)
    assert output["advantages"].shape == (8, 7)
    assert output["advantages"][0].tolist() == [0.25, 0.25, 0, -0.25, -0.25, 0, 0]


@pytest.mark.parametrize(
    "flag,mode",
    [
        (False, "strict_progress_v1"),
        (None, "strict_progress_v1"),
        (True, "legacy"),
    ],
)
def test_composed_trainer_mode_must_match_actor_reward_mode(flag, mode):
    cls = production_method(
        "CappedPPOTrainerSync",
        "_procredit_runtime",
        object,
        os=SimpleNamespace(environ={"AGENTIC_RL_CONFIG": "runtime.yaml"}),
        load_runtime_config=lambda path: {"reward": {"mode": mode}},
        is_procredit=lambda cfg: cfg["reward"]["mode"] == "strict_progress_v1",
        CreditConfig=CreditConfig,
        Path=Path,
    )
    trainer = cls()
    trainer.config = SimpleNamespace(
        algorithm={} if flag is None else {"procredit_enabled": flag},
        trainer=SimpleNamespace(default_local_dir="run/checkpoints"),
    )
    with pytest.raises(ValueError, match="mode"):
        trainer._procredit_runtime()
