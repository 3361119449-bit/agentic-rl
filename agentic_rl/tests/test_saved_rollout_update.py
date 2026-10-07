"""Saved tokens must reproduce every policy prefix and retain frozen advantages."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


def fixture():
    mapping = [0, 0, -1, -1, 1, -1, -1]
    record = {
        "schema_version": "2.0",
        "trajectory_id": "a",
        "task_id": "task",
        "split": "train",
        "policy_version": 0,
        "assistant_turns": 2,
        "trajectory_tokens": 9,
        "response_turn_ids": mapping,
        "user_sim_result": {"user_sim_valid": True},
        "token_turns": [
            {
                "assistant_turn_index": 0,
                "prompt_token_ids": [10, 11],
                "output_token_ids": [12, 13],
                "output_old_log_probs": [-0.1, -0.2],
            },
            {
                "assistant_turn_index": 1,
                "prompt_token_ids": [10, 11, 12, 13, 90, 91],
                "output_token_ids": [14],
                "output_old_log_probs": [-0.3],
            },
        ],
    }
    row = {
        "trajectory_id": "a",
        "task_id": "task",
        "policy_version": 0,
        "response_turn_ids": mapping,
    }
    credit = {"trajectory_id": "a", "token_advantages": [0.25, 0.25, 0, 0, -0.25, 0, 0]}
    return record, row, credit


def test_restore_retains_observation_context_and_all_policy_tokens():
    from scripts.saved_rollout_support.batch import restore_trajectory

    result = restore_trajectory(*fixture())
    assert result.prompt_ids == [10, 11]
    assert result.response_ids == [12, 13, 90, 91, 14]
    assert result.response_mask == [1, 1, 0, 0, 1]
    assert result.old_log_probs == [-0.1, -0.2, 0, 0, -0.3]
    assert result.advantages == [0.25, 0.25, 0, 0, -0.25]
    assert result.trimmed_tail_tokens == 2


@pytest.mark.parametrize(
    "damage",
    [
        "prefix",
        "output",
        "mapping",
        "missing_lp",
        "nan_lp",
        "policy_tail",
        "adv_tail",
        "version",
        "invalid_user",
    ],
)
def test_restore_rejects_data_that_would_change_the_ppo_batch(damage):
    from scripts.saved_rollout_support.batch import restore_trajectory

    record, row, credit = deepcopy(fixture())
    if damage == "prefix":
        record["token_turns"][1]["prompt_token_ids"][0] = 99
    elif damage == "output":
        record["token_turns"][0]["output_token_ids"][0] = 99
    elif damage == "mapping":
        record["response_turn_ids"][2] = 0
    elif damage == "missing_lp":
        record["token_turns"][0]["output_old_log_probs"].pop()
    elif damage == "nan_lp":
        record["token_turns"][0]["output_old_log_probs"][0] = math.nan
    elif damage == "policy_tail":
        record["response_turn_ids"][-1] = 1
    elif damage == "adv_tail":
        credit["token_advantages"][-1] = 0.5
    elif damage == "version":
        record["policy_version"] = row["policy_version"] = 1
    else:
        record["user_sim_result"]["user_sim_valid"] = False
    with pytest.raises(ValueError):
        restore_trajectory(record, row, credit)


def group_report():
    from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
    from tau2_agentic_rl.versions import sha256_json

    row = {
        "trajectory_id": "a",
        "task_id": "task",
        "policy_version": 0,
        "checkset_fingerprint": "checks",
        "initial_state_fingerprint": "initial",
        "score": 1.5,
        "valid": True,
        "terminal_success": 1,
        "multiplier": 1,
        "phi": [0, 1, 1],
        "credit_version": "procredit-turn-v4",
        "response_turn_ids": [0, 0, -1, -1, 1, -1, -1],
        "policy_credit": {
            "version": "policy-credit-v1",
            "violating_turns": [],
            "attribution_complete": True,
            "unresolved_checks": [],
        },
        "process_credit": {"version": "process-credit-v2", "turn_costs": [0, 0]},
    }
    report = {
        "version": "procredit-group-v2",
        "uid": "u",
        "members": ["u_0_0"],
        "inputs": [row],
        "inputs_fingerprint": sha256_json([row]),
        "credit": compute_group_credit(
            [row], CreditConfig(version="procredit-turn-v4")
        ),
        "decision": "has_signal",
        "terminal_group": {
            "status": "failure",
            "rollout_n": 2,
            "failed_session_ids": [1],
            "failure_kind": "transient_exhausted",
        },
    }
    report["audit_fingerprint"] = sha256_json(report)
    return report


def test_group_preserves_salvaged_members_without_inventing_failed_trajectories():
    from scripts.saved_rollout_support.batch import validate_group

    result = validate_group(group_report(), group_size=2)
    assert len(result) == 1
    assert result[0][1]["token_advantages"] == [0.25, 0.25, 0, 0, -0.25, 0, 0]


@pytest.mark.parametrize("damage", ["advantage", "slot", "input_hash", "fingerprint"])
def test_group_rejects_changed_advantages_or_membership_even_if_resigned(damage):
    from scripts.saved_rollout_support.batch import validate_group
    from tau2_agentic_rl.versions import sha256_json

    report = group_report()
    if damage == "advantage":
        report["credit"]["trajectories"][0]["token_advantages"][0] = 100
    elif damage == "slot":
        report["members"] = ["u_1_0"]
    elif damage == "input_hash":
        report["inputs_fingerprint"] = "changed"
    else:
        report["audit_fingerprint"] = "changed"
    if damage != "fingerprint":
        report["audit_fingerprint"] = sha256_json(
            {k: v for k, v in report.items() if k != "audit_fingerprint"}
        )
    with pytest.raises(ValueError):
        validate_group(report, group_size=2)


def test_new_cli_help_does_not_import_gpu_or_generation_dependencies():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "update_saved_rollouts.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "--results" in completed.stdout
    assert "--dry-run" in completed.stdout


def test_import_of_materializer_needs_no_ray_vllm_or_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.saved_rollout_support.batch; assert not {'ray', 'vllm', 'torch'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_launch_rejects_an_adapter_that_did_not_generate_version_zero():
    from scripts.saved_rollout_support.runtime import replay_overrides

    command = [
        "python",
        "-m",
        "tau2_agentic_rl.verl_entrypoint",
        "actor_rollout_ref.model.lora_adapter_path=/different/policy",
    ]
    with pytest.raises(ValueError):
        replay_overrides(command, Path("model"), Path("output"))


def test_launch_keeps_effective_training_settings_and_relocates_only_outputs():
    from scripts.saved_rollout_support.runtime import replay_overrides

    command = [
        "python",
        "-m",
        "tau2_agentic_rl.verl_entrypoint",
        "actor_rollout_ref.actor.optim.lr=5e-6",
        "actor_rollout_ref.actor.ppo_epochs=2",
        "trainer.save_freq=10",
        "trainer.save_freq=1",
        "trainer.default_local_dir=old/checkpoints",
    ]
    result = replay_overrides(command, Path("model"), Path("new-output"))
    values = {
        item.lstrip("+").partition("=")[0]: item.partition("=")[2] for item in result
    }
    assert values["actor_rollout_ref.actor.optim.lr"] == "5e-6"
    assert values["actor_rollout_ref.actor.ppo_epochs"] == "2"
    assert values["trainer.save_freq"] == "1"
    assert values["actor_rollout_ref.model.path"] == json.dumps(
        str(Path("model").resolve())
    )
    assert values["trainer.resume_mode"] == "disable"
    assert values["trainer.total_training_steps"] == "1"


def test_output_directory_cannot_replace_the_original_run(scratch_dir):
    from scripts.saved_rollout_support.runtime import reserve_output

    (scratch_dir / "checkpoint").write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        reserve_output(scratch_dir)
    assert (scratch_dir / "checkpoint").read_text(encoding="utf-8") == "original"


def test_directory_reservation_is_exclusive_even_before_checkpoint_save(scratch_dir):
    from scripts.saved_rollout_support.runtime import reserve_output

    output = scratch_dir / "new-run"
    reserve_output(output)
    with pytest.raises(FileExistsError):
        reserve_output(output)


def test_matching_base_with_adapter_artifacts_cannot_load_a_different_policy(
    scratch_dir,
):
    from scripts.saved_rollout_support.batch import SavedBatch, verify_base_model
    from tau2_agentic_rl.base_identity import capture_base_identity

    model = scratch_dir / "base"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text(
        '{"chat_template": "original"}', encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"original weights")
    batch = SavedBatch([], {}, capture_base_identity(model), {})
    (model / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "another-model"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="adapter"):
        verify_base_model(batch, model)


@pytest.mark.parametrize(
    "grad_norms",
    [[0.1, float("nan"), 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, float("inf"), 0.2, 0.3]],
)
def test_skipped_or_missing_optimizer_iterations_cannot_count_as_complete(grad_norms):
    from scripts.saved_rollout_support.runtime import validate_optimizer_metrics

    with pytest.raises(ValueError):
        validate_optimizer_metrics(
            {"grad_norm": grad_norms, "loss": [0.1] * 4}, expected_iterations=4
        )


def test_all_optimizer_iterations_have_finite_loss_and_gradients():
    from scripts.saved_rollout_support.runtime import validate_optimizer_metrics

    assert (
        validate_optimizer_metrics(
            {"grad_norm": [[0.1], [0.2], [0.3], [0.4]], "loss": [0.1] * 4},
            expected_iterations=4,
        )
        == 4
    )
