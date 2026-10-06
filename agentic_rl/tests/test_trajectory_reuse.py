"""Keep genuine members, reject missing evidence, and exclude synthetic padding."""

from copy import deepcopy

import pytest
from test_turn_local_v4 import rows

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.procredit_runtime import build_credit_tensors, queue_group_reports

CFG = CreditConfig(version="procredit-turn-v4")


def partial_group(failed=(7,)):
    kept = [i for i in range(8) if i not in failed]
    group = rows()
    return (
        [f"group_{i}_0" for i in kept],
        [{"procredit": group[i]} for i in kept],
        {
            "group": {
                "status": "failure",
                "rollout_n": 8,
                "failure_kind": "transient_exhausted",
                "failed_session_ids": list(failed),
            }
        },
    )


def test_partial_group_keeps_seven_members_and_negative_policy_credit(scratch_dir):
    keys, extras, context = partial_group()
    report = queue_group_reports(
        keys, extras, CFG, terminal_groups=context, audit_dir=scratch_dir
    )["group"]
    assert report["members"] == [f"group_{i}_0" for i in range(7)]
    assert report["terminal_group"]["failed_session_ids"] == [7]
    assert report["credit"]["has_signal"]
    assert all(
        t["token_advantages"] == [0, 0, 0, 0, -1]
        for t in report["credit"]["trajectories"]
    )


def test_singleton_retains_local_signal_without_inventing_scalar_advantage():
    keys, extras, context = partial_group(tuple(range(1, 8)))
    credit = queue_group_reports(keys, extras, CFG, terminal_groups=context)["group"][
        "credit"
    ]
    assert credit["has_signal"]
    assert credit["trajectories"][0]["trajectory_advantage"] == 0
    assert credit["trajectories"][0]["token_advantages"] == [0, 0, 0, 0, -1]


def test_singleton_without_real_signal_stays_filtered():
    group = rows()[:1]
    group[0]["valid"] = True
    group[0]["policy_credit"]["violating_turns"] = []
    assert not compute_group_credit(group, CFG)["has_signal"]


@pytest.mark.parametrize(
    "change",
    [
        "fatal",
        "missing_member",
        "failed_output",
        "duplicate_slot",
        "wrong_n",
        "finished_failure",
    ],
)
def test_partial_group_never_treats_corruption_as_slot_exhaustion(change):
    keys, extras, context = partial_group()
    tag = context["group"]
    if change == "fatal":
        tag["failure_kind"] = "fatal"
    elif change == "missing_member":
        keys, extras = keys[:-1], extras[:-1]
    elif change == "failed_output":
        keys.append("group_7_0")
        extras.append({"procredit": rows()[7]})
    elif change == "duplicate_slot":
        keys[-1] = "group_0_1"
    elif change == "wrong_n":
        tag["rollout_n"] = 7
    else:
        tag["status"] = "finished"
    with pytest.raises(ValueError):
        queue_group_reports(keys, extras, CFG, terminal_groups=context)


def test_partial_group_requires_explicit_terminal_session_evidence():
    keys, extras, _ = partial_group()
    with pytest.raises(ValueError, match="incomplete"):
        queue_group_reports(keys, extras, CFG)


@pytest.mark.parametrize("nested", [False, True])
def test_padding_copied_from_real_row_does_not_change_credit_or_gradient(nested):
    torch = pytest.importorskip("torch")
    keys, extras, context = partial_group()
    keys += ["padabc_0_0"]
    extras += [deepcopy(extras[0])]  # Pinned veRL copies a real row's extra_fields.
    padding = [False] * 7 + [True]
    masks = [torch.tensor([1, 1, 0, 1, 1]) for _ in range(7)]
    masks.append(torch.tensor([0] if nested else [0] * 5))
    mask = (
        torch.nested.as_nested_tensor(masks, layout=torch.jagged)
        if nested
        else torch.stack(masks)
    )
    fields, metrics = build_credit_tensors(
        keys, extras, mask, CFG, terminal_groups=context, padding=padding
    )
    advantages = list(fields["advantages"].unbind())
    assert advantages[0].tolist() == [0, 0, 0, 0, -1]
    assert torch.count_nonzero(advantages[-1]) == 0
    assert metrics["procredit/real_trajectories"] == 7
    assert metrics["procredit/padding_rows"] == 1
    assert metrics["procredit/policy_violation_turns"] == 7
    current = [torch.zeros_like(a, requires_grad=True) for a in advantages]
    loss = -sum(
        (a * logp.exp() * m).sum()
        for a, logp, m in zip(advantages, current, masks, strict=True)
    )
    loss.backward()
    assert current[0].grad[-1] > 0
    assert torch.count_nonzero(current[-1].grad) == 0


def test_padding_with_trainable_tokens_is_fatal():
    torch = pytest.importorskip("torch")
    keys, extras, context = partial_group()
    keys += ["padabc_0_0"]
    extras += [extras[0]]
    mask = torch.tensor([[1, 1, 0, 1, 1]] * 8)
    with pytest.raises(ValueError, match="padding"):
        build_credit_tensors(
            keys,
            extras,
            mask,
            CFG,
            terminal_groups=context,
            padding=[False] * 7 + [True],
        )


def test_partial_group_replays_with_exact_membership_and_audit(scratch_dir):
    from test_procredit_config import ROOT

    from scripts.rescore_procredit_groups import rescore_groups
    from tau2_agentic_rl.config import load_yaml

    keys, extras, context = partial_group()
    source, output = scratch_dir / "source", scratch_dir / "output"
    queue_group_reports(keys, extras, CFG, terminal_groups=context, audit_dir=source)
    config = load_yaml(ROOT / "configs/rl/airline_procredit_v4.yaml")
    assert rescore_groups(source, output, config) == 1
    assert (
        next(source.glob("*.json")).read_bytes()
        == next(output.glob("*.json")).read_bytes()
    )


def test_trainer_preserves_balanced_batch_metadata_after_writing_credit():
    from types import SimpleNamespace

    from test_procredit_runtime import production_method

    torch = pytest.importorskip("torch")
    td = pytest.importorskip("tensordict")
    keys, extras, context = partial_group()
    # Reordering during balancing can place synthetic rows before real members.
    keys = [f"padabc_{i}_0" for i in range(25)] + list(reversed(keys))
    extras = [deepcopy(extras[0]) for _ in range(25)] + list(reversed(extras))
    tags = [{"is_padding": True}] * 25 + [{"is_padding": False}] * 7
    mask = torch.tensor([[0] * 5] * 25 + [[1, 1, 0, 1, 1]] * 7)
    original_old = torch.randn(32, 5) * mask
    stored = {
        "response_mask": mask,
        "extra_fields": extras,
        "old_log_probs": original_old,
    }
    batch = SimpleNamespace(
        keys=keys,
        tags=tags,
        partition_id="train",
        extra_info={"procredit_terminal_groups": context, "temperature": 1.0},
    )

    def write(**kwargs):
        stored.update(kwargs["fields"])
        return SimpleNamespace(keys=keys, tags=[{}] * 32, extra_info={})

    trainer = production_method(
        "CappedPPOTrainerSync",
        "_compute_advantage",
        object,
        tq=SimpleNamespace(kv_batch_get=lambda **kw: stored, kv_batch_put=write),
        TensorDict=td.TensorDict,
    )()
    trainer._procredit_runtime = lambda: (CFG, None)
    metrics = {}
    result = trainer._compute_advantage(batch, metrics)
    assert result is batch
    assert stored["old_log_probs"] is original_old
    assert metrics["procredit/policy_violation_turns"] == 7
    assert metrics["procredit/padding_rows"] == 25
    assert torch.count_nonzero(stored["advantages"][:25]) == 0
    assert torch.all(stored["advantages"][25:, -1] == -1)
