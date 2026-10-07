"""Actor-only veRL update; imported exclusively by the saved-rollout entrypoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from scripts.saved_rollout_support.batch import VERL_COMMIT, SavedBatch


def replay_overrides(command, model_path: Path, output_dir: Path):
    if command[1:3] != ["-m", "tau2_agentic_rl.verl_entrypoint"]:
        raise ValueError("unsupported saved launcher")
    overrides = {}
    for item in command[3:]:
        key, separator, value = item.partition("=")
        if not separator or key.startswith("~"):
            raise ValueError("unsupported saved launch override")
        normalized = key.lstrip("+")
        if (
            normalized
            in {
                "actor_rollout_ref.model.lora_adapter_path",
                "actor_rollout_ref.model.lora.adapter_path",
            }
            and value != "null"
        ):
            raise ValueError(
                "version 0 replay requires a fresh adapter on the original SFT base"
            )
        # Keep the final occurrence, including Hydra's + prefix for new keys.
        overrides[normalized] = (key, value)
    replacements = {
        "actor_rollout_ref.model.path": json.dumps(str(model_path.resolve())),
        "trainer.default_local_dir": json.dumps(
            str(output_dir.resolve() / "checkpoints")
        ),
        "trainer.rollout_data_dir": json.dumps(
            str(output_dir.resolve() / "unused_rollouts")
        ),
        "trainer.resume_mode": "disable",
        "trainer.resume_from_path": "null",
        "trainer.total_training_steps": "1",
        "trainer.logger": "[console]",
        "trainer.val_before_train": "false",
        "trainer.test_freq": "-1",
    }
    for key, value in replacements.items():
        overrides[key] = (key, value)
    return [f"{key}={value}" for key, value in overrides.values()]


def reserve_output(output_dir: Path):
    """A new directory is also the reservation against concurrent updates."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)


def build_tensor_batch(batch: SavedBatch, *, eos_token_id: int):
    import torch
    from tensordict import TensorDict

    rows = list(batch.trajectories)
    from scripts.saved_rollout_support.batch import SavedTrajectory

    rows.extend(
        SavedTrajectory(
            f"padding-{i}", [eos_token_id], [eos_token_id], [0], [0.0], [0.0], 0
        )
        for i in range(batch.manifest["padding_rows"])
    )

    def nested(values, dtype):
        return torch.nested.as_nested_tensor(
            [torch.tensor(v, dtype=dtype) for v in values], layout=torch.jagged
        )

    masks = nested([r.response_mask for r in rows], torch.int64)
    return TensorDict(
        {
            "input_ids": nested(
                [r.prompt_ids + r.response_ids for r in rows], torch.int64
            ),
            "position_ids": nested(
                [list(range(len(r.prompt_ids) + len(r.response_ids))) for r in rows],
                torch.int64,
            ),
            "prompts": nested([r.prompt_ids for r in rows], torch.int64),
            "responses": nested([r.response_ids for r in rows], torch.int64),
            "response_mask": masks,
            "loss_mask": masks,
            "old_log_probs": nested([r.old_log_probs for r in rows], torch.float32),
            "advantages": nested([r.advantages for r in rows], torch.float32),
        },
        batch_size=[len(rows)],
    )


def _digest_tensor(digest, tensor):
    import torch

    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    value = tensor.detach().cpu().contiguous()
    digest.update(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def frozen_batch_digest(data):
    digest = hashlib.sha256()
    for key in ("old_log_probs", "advantages", "response_mask"):
        digest.update(key.encode())
        _digest_tensor(digest, data[key].values())
        _digest_tensor(digest, data[key].offsets())
    return digest.hexdigest()


def trainable_parameter_digest(module):
    digest, count = hashlib.sha256(), 0
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            _digest_tensor(digest, parameter)
            count += 1
    if not count:
        raise ValueError("actor has no trainable parameters")
    return digest.hexdigest()


def validate_optimizer_metrics(metrics, *, expected_iterations):
    def flatten(values):
        if isinstance(values, (list, tuple)):
            return [v for value in values for v in flatten(value)]
        if hasattr(values, "item"):
            values = values.item()
        if not isinstance(values, (int, float)) or not math.isfinite(values):
            raise ValueError(
                "non-finite optimizer loss or gradient; update may have been skipped"
            )
        return [values]

    if not isinstance(metrics, dict):
        raise ValueError("actor did not return optimizer metrics")
    for key in ("grad_norm", "loss"):
        if key not in metrics or len(flatten(metrics[key])) != expected_iterations:
            raise ValueError(
                f"actor did not report all {expected_iterations} optimizer iterations: {key}"
            )
    return expected_iterations


def make_actor_worker():
    from verl.single_controller.base.decorator import (
        make_nd_compute_dataproto_dispatch_fn,
        register,
    )
    from verl.workers.engine_workers import ActorRolloutRefWorker

    class SavedRolloutActorWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn("actor"))
        def update_saved_batch(self, data):
            from tensordict import NonTensorData
            from verl.utils import tensordict_utils as tu

            mini_batch = tu.get_non_tensor_data(data, "mini_batch_size", None)
            epochs = tu.get_non_tensor_data(data, "epochs", None)
            iterations = data.shape[0] // mini_batch * epochs
            before = frozen_batch_digest(data)
            parameters_before = trainable_parameter_digest(self.actor.engine.module)
            output = super().update_actor(data)
            validate_optimizer_metrics(
                tu.get_non_tensor_data(output, "metrics", None),
                expected_iterations=iterations,
            )
            after = frozen_batch_digest(data)
            parameters_after = trainable_parameter_digest(self.actor.engine.module)
            if before != after:
                raise ValueError(
                    "worker changed frozen old probabilities, advantages or response mask"
                )
            if parameters_before == parameters_after:
                raise ValueError(
                    "PPO returned without changing any trainable actor parameters"
                )
            output["replay_worker_audit"] = NonTensorData(
                {
                    "frozen_batch_sha256_before": before,
                    "frozen_batch_sha256_after": after,
                    "trainable_parameters_sha256_before": parameters_before,
                    "trainable_parameters_sha256_after": parameters_after,
                    "trainable_parameters_changed": True,
                    "observed_optimizer_iterations": iterations,
                }
            )
            return output

    return SavedRolloutActorWorker


def compose_actor_config(batch, model_path, output_dir, verl_root):
    import hydra
    from omegaconf import open_dict

    overrides = replay_overrides(batch.launch["command"], model_path, output_dir)
    with hydra.initialize_config_dir(
        config_dir=str((verl_root / "verl" / "trainer" / "config").resolve()),
        version_base=None,
    ):
        config = hydra.compose(config_name="ppo_trainer", overrides=overrides)
    actor, model = config.actor_rollout_ref.actor, config.actor_rollout_ref.model
    if (
        config.trainer.nnodes != 1
        or config.trainer.n_gpus_per_node != 1
        or config.algorithm.adv_estimator != "grpo"
        or config.algorithm.use_kl_in_reward
        or actor.use_kl_loss
        or not model.use_remove_padding
        or actor.strategy not in {"fsdp", "fsdp2"}
        or model.lora_rank <= 0
        or model.lora_adapter_path is not None
        or model.lora.get("adapter_path") is not None
        or model.hf_config_path is not None
        or model.tokenizer_path is not None
        or model.custom_chat_template is not None
        or config.get("distillation", {}).get("enable", False)
        or not config.algorithm.rollout_correction.bypass_mode
        or config.algorithm.rollout_correction.loss_type != "ppo_clip"
        or actor.policy_loss.get("loss_mode", "vanilla") != "vanilla"
    ):
        raise ValueError(
            "saved launch differs from supported single-GPU, initial-LoRA, GRPO PPO flow"
        )
    mini_batch = actor.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n
    if (
        mini_batch != batch.manifest["ppo_mini_batch_size"]
        or actor.ppo_epochs != batch.manifest["ppo_epochs"]
    ):
        raise ValueError(
            "effective actor batch configuration differs from frozen launch"
        )
    # The normal PPO trainer sets this when it creates its dataloader. We do not
    # create that dataloader: this entrypoint performs exactly one outer update.
    with open_dict(config):
        actor.optim.total_training_steps = 1
    return config


def _write_json(path, value):
    def scalar(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"non-serializable update metric: {type(item).__name__}")

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=scalar) + "\n",
        encoding="utf-8",
    )


def run_update(
    batch: SavedBatch, *, model_path: Path, verl_root: Path, output_dir: Path
):
    from tau2_agentic_rl.base_identity import save_base_identity
    from tau2_agentic_rl.ppo_audit import audit_update

    actual = subprocess.run(
        ["git", "-C", str(verl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != VERL_COMMIT:
        raise ValueError(
            f"veRL checkout mismatch: expected {VERL_COMMIT}, got {actual}"
        )
    if not batch.manifest["base_model_verified"]:
        raise ValueError(
            "original base-model contents must be verified before actor initialization"
        )
    sys.path.insert(0, str(verl_root.resolve()))
    import verl

    if Path(verl.__file__).resolve().parent != verl_root.resolve() / "verl":
        raise ValueError("Python imported a different veRL checkout")
    config = compose_actor_config(batch, model_path, output_dir, verl_root)
    import ray
    import torch
    from omegaconf import OmegaConf
    from verl.single_controller.ray import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import response_from_nested

    if not torch.cuda.is_available():
        raise RuntimeError("actual saved-rollout PPO update requires CUDA")
    output_dir = output_dir.resolve()
    reserve_output(output_dir)
    save_base_identity(output_dir, batch.base_identity)
    _write_json(output_dir / "replay_manifest.json", batch.manifest)
    OmegaConf.save(config, output_dir / "actor_config.yaml", resolve=True)
    status = {"status": "initializing_actor", "outer_update_steps": 0}
    _write_json(output_dir / "update_status.json", status)
    owned_ray = not ray.is_initialized()
    try:
        if owned_ray:
            ray.init(
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": os.pathsep.join(sys.path),
                        "TOKENIZERS_PARALLELISM": "false",
                    }
                }
            )
        pool = RayResourcePool(
            process_on_nodes=[1],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="saved_rollout_actor",
        )
        wrapper = RayClassWithInitArgs(
            cls=ray.remote(make_actor_worker()),
            config=config.actor_rollout_ref,
            role="actor",
        )
        workers = RayWorkerGroup(
            resource_pool=pool, ray_cls_with_init=wrapper, device_name="cuda"
        )
        workers.init_model()
        # A token already present in the verified original prompt is sufficient
        # for the fully masked dummy rows; no tokenizer or generation is needed.
        data = build_tensor_batch(
            batch, eos_token_id=batch.trajectories[0].prompt_ids[0]
        )
        actor = config.actor_rollout_ref.actor

        def read_old():
            return (
                [row.tolist() for row in data["old_log_probs"].unbind()],
                [row.tolist() for row in data["response_mask"].unbind()],
            )

        def compute_current():
            probe = data.clone()
            tu.assign_non_tensor(
                probe,
                calculate_entropy=False,
                compute_loss=False,
                temperature=config.actor_rollout_ref.rollout.temperature,
            )
            output = workers.compute_log_prob(probe)
            values = response_from_nested(output["log_probs"], data["response_mask"])
            return [row.tolist() for row in values.unbind()]

        def update():
            status["status"] = "update_started"
            _write_json(output_dir / "update_status.json", status)
            tu.assign_non_tensor(
                data,
                calculate_entropy=actor.calculate_entropy or actor.entropy_coeff != 0,
                distillation_use_topk=False,
                distillation_only=False,
                global_batch_size=batch.manifest["ppo_mini_batch_size"],
                mini_batch_size=batch.manifest["ppo_mini_batch_size"],
                epochs=actor.ppo_epochs,
                seed=actor.data_loader_seed,
                dataloader_kwargs={"shuffle": actor.shuffle},
                temperature=config.actor_rollout_ref.rollout.temperature,
            )
            return workers.update_saved_batch(data)

        output, audit = audit_update(
            read_old=read_old,
            compute_current=compute_current,
            update=update,
            path=output_dir / "ppo_audit.json",
        )
        worker_audit = tu.get_non_tensor_data(output, "replay_worker_audit", None)
        metrics = tu.get_non_tensor_data(output, "metrics", None)
        _write_json(output_dir / "worker_audit.json", worker_audit)
        _write_json(output_dir / "metrics.json", metrics)
        checkpoint = output_dir / "checkpoints" / "global_step_1" / "actor"
        status.update(status="saving_checkpoint", outer_update_steps=1)
        _write_json(output_dir / "update_status.json", status)
        workers.save_checkpoint(
            local_path=str(checkpoint),
            hdfs_path=None,
            global_step=1,
            max_ckpt_to_keep=None,
        )
        save_base_identity(checkpoint, batch.base_identity)
        status.update(
            status="complete",
            checkpoint=str(checkpoint),
            trainable_parameters_changed=worker_audit["trainable_parameters_changed"],
            observed_optimizer_iterations=worker_audit["observed_optimizer_iterations"],
            expected_optimizer_iterations=batch.manifest[
                "expected_optimizer_iterations"
            ],
        )
        _write_json(output_dir / "update_status.json", status)
        print(json.dumps(status, indent=2))
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        _write_json(output_dir / "update_status.json", status)
        raise
    finally:
        if owned_ray and ray.is_initialized():
            ray.shutdown()
