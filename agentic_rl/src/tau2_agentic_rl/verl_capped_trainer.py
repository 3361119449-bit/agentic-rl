"""Pinned veRL v0.9.0 trainer extensions for bounded dynamic sampling.

veRL's V1 replay buffer intentionally ignores ``max_num_gen_batches``. This
extension bounds sampling to three waves of eight prompts. ProCredit v4 consumes
all usable groups, including verified partial groups and a sub-target remainder.
Legacy sampling retains its fixed four-group contract. It deliberately
uses veRL V1 protected APIs and must only be used with the commit checked by the
launcher.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.v1.replay_buffer import (
    DAPO_FILTERED_REWARD_COUNTS_KEY,
    ReplayBuffer,
    _accumulate_eviction_metrics,
)
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.debug import marked_timer
from verl.utils.skip import SkipManager
from verl.utils.tracking import (
    DapoFilteredRewardTableLogger,
    Tracking,
    ValidationGenerationsLogger,
)

from tau2_agentic_rl.advantages import CreditConfig, is_procredit
from tau2_agentic_rl.checkpoints import restore_step_clock
from tau2_agentic_rl.concurrency import api_budget
from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.dynamic_sampling import TrainingStepClock
from tau2_agentic_rl.failures import raise_if_fatal
from tau2_agentic_rl.ppo_audit import audit_update
from tau2_agentic_rl.procredit_runtime import (
    _save_group_audit,
    build_credit_tensors,
    queue_group_reports,
    terminal_group_contract,
)
from tau2_agentic_rl.rl_resume import snapshot_resume_identity

logger = logging.getLogger(__name__)


@dataclass
class DynamicSamplingCapReached(RuntimeError):
    """Signal that an attempted optimizer batch exhausted its rollout budget."""

    generated_prompt_groups: int
    generated_trajectories: int
    metrics: dict

    def __str__(self) -> str:
        return (
            "dynamic-sampling cap reached after "
            f"{self.generated_prompt_groups} prompt groups / "
            f"{self.generated_trajectories} logical rollout slots"
        )


class CappedDynamicReplayBuffer(ReplayBuffer):
    """Synchronous group filter with a hard logical-generation-batch cap."""

    def __init__(
        self,
        *args,
        conceptual_gen_batch_size: int,
        max_num_gen_batches: int,
        rollout_group_size: int,
        credit_config: CreditConfig | None = None,
        group_audit_dir: Path | None = None,
        rollout_counts_fn=None,
        fatal_error_fn=None,
        terminal_metadata_fn=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            conceptual_gen_batch_size <= 0
            or max_num_gen_batches <= 0
            or rollout_group_size <= 1
        ):
            raise ValueError(
                "bounded dynamic-sampling sizes must be positive, and rollout group size > 1"
            )
        self.conceptual_gen_batch_size = conceptual_gen_batch_size
        self.max_num_gen_batches = max_num_gen_batches
        self.rollout_group_size = rollout_group_size
        self.credit_config = credit_config
        self.group_audit_dir = group_audit_dir
        self.credit_cache = {}
        self.rollout_counts_fn = rollout_counts_fn
        self.fatal_error_fn = fatal_error_fn
        self.terminal_metadata_fn = terminal_metadata_fn or tq.kv_list
        self.terminal_groups = {}
        self.rollout_counts_start = rollout_counts_fn() if rollout_counts_fn else None

    def begin_attempt(self):
        """Snapshot before the trainer submits even the first logical batch."""
        self._check_fatal_rollout()
        if self.rollout_counts_fn is not None:
            self.rollout_counts_start = self.rollout_counts_fn()

    def _check_fatal_rollout(self):
        check = getattr(self, "fatal_error_fn", None)
        if check is not None:
            raise_if_fatal(check())

    def _sync_metadata_from_transfer_queue(self):
        # The parent's validation sampler also polls through this method.
        self._check_fatal_rollout()
        super()._sync_metadata_from_transfer_queue()
        self._check_fatal_rollout()
        if self._isolates_failed_groups("train"):
            # The pinned parent keeps status but drops session membership. Read
            # terminal tags only after the parent observed the settlement barrier.
            metadata = self.terminal_metadata_fn() or {}
            terminal = self.finished_keys["train"] | self.failure_keys["train"]
            groups = {
                uid: terminal_group_contract(metadata.get("train", {}).get(uid), self.rollout_group_size)
                for uid in terminal
            }
            for uid, tag in groups.items():
                if (tag["status"] == "failure") != (uid in self.failure_keys["train"]):
                    raise ValueError("terminal group status changed after settlement")
            self.terminal_groups = {"train": groups}
            # Postprocessing can write an output and then raise. Its failed
            # session must never masquerade as a scored successful sibling.
            rejected = []
            for key in self.partitions["train"]:
                parts = key.split("_")
                if parts[0] in groups:
                    if len(parts) != 3 or not all(p.isdigit() for p in parts[1:]):
                        raise ValueError("invalid terminal session queue key")
                    if int(parts[1]) in groups[parts[0]]["failed_session_ids"]:
                        rejected.append(key)
            if rejected:
                tq.kv_clear(keys=rejected, partition_id="train")
                for key in rejected:
                    del self.partitions["train"][key]
            self._check_fatal_rollout()

    def _isolates_failed_groups(self, partition_id):
        return (
            partition_id == "train"
            and getattr(getattr(self, "credit_config", None), "version", None) == "procredit-turn-v4"
        )

    def _audit_exhausted_groups(self, partition_id, uids):
        if self.group_audit_dir is None:
            return
        directory = self.group_audit_dir / "exhausted"
        for uid in sorted(uids):
            report = {
                "uid": uid, "status": "transient_exhausted", "reason": "failed_rollout_slot",
                "terminal_group": self.terminal_groups[partition_id][uid],
                "materializable_members": sorted(
                    key for key in self.partitions[partition_id] if key.split("_")[0] == uid
                ),
                "policy_version": self.prompt_global_steps[partition_id][uid],
            }
            _save_group_audit(directory, report)

    def _dapo_filtered_keys(self, partition_id: str):
        if self.credit_config is None:
            return super()._dapo_filtered_keys(partition_id)
        if partition_id != "train":
            return set(), Counter()
        finished = set(self.finished_keys[partition_id])
        local_only = self.credit_config.version == "procredit-turn-v4"
        terminal_groups = None
        if local_only:
            finished |= self.failure_keys[partition_id]
            terminal_groups = self.terminal_groups[partition_id]
            # Empty exhausted groups cannot yield any credit. Missing rows in a
            # finished/partially successful group are corruption and still fail.
            finished -= {uid for uid, tag in terminal_groups.items()
                         if len(tag["failed_session_ids"]) == tag["rollout_n"]}
        self.credit_cache = {
            uid: report for uid, report in self.credit_cache.items() if uid in finished
        }
        missing = finished - self.credit_cache.keys()
        keys = sorted(
            key for key in self.partitions[partition_id]
            if key.split("_")[0] in missing
        )
        if missing:
            if {key.split("_")[0] for key in keys} != missing:
                raise ValueError("finished ProCredit group has no materializable rows")
            data = tq.kv_batch_get(
                keys=keys, partition_id=partition_id, select_fields=["extra_fields"]
            )
            self.credit_cache.update(queue_group_reports(
                keys, list(data["extra_fields"]), self.credit_config,
                audit_dir=self.group_audit_dir,
                group_size=getattr(self, "rollout_group_size", 8),
                terminal_groups=terminal_groups,
            ))
        filtered = {
            uid: report["credit"]["score_mean"]
            for uid, report in self.credit_cache.items()
            if not report["credit"]["has_signal"]
        }
        return set(filtered), Counter(filtered.values())

    def _clear_attempt(self, partition_id: str) -> None:
        """Remove every prompt and trajectory left by the capped attempt."""
        self._sync_metadata_from_transfer_queue()
        all_prompt_uids = set(self.prompt_global_steps[partition_id])
        self._clear_groups(partition_id, all_prompt_uids)

    def _terminal_eviction_reasons(self, global_steps: int, partition_id: str):
        reasons = super()._terminal_eviction_reasons(global_steps, partition_id)
        if partition_id != "train":
            return reasons
        stale, constant, failed, counts = reasons
        if self._isolates_failed_groups(partition_id):
            # A failed slot does not invalidate its scored siblings. The parent
            # evicts only empty failed groups; local-credit filtering handles
            # signal-free survivors using their actual membership.
            return reasons
        # The pinned parent only evicts failed groups with zero materializable
        # trajectories. Our GRPO contract also excludes 7-success + 1-API-error
        # groups, even if their surviving rewards have variance.
        return stale, constant, failed | set(self.failure_keys[partition_id]), counts

    def _add_sampling_metrics(
        self,
        metrics: dict,
        generated_batches: int,
        valid_groups: int,
    ) -> None:
        generated_groups = generated_batches * self.conceptual_gen_batch_size
        filtered = metrics.get(DAPO_FILTERED_REWARD_COUNTS_KEY, {})
        metrics.update(
            {
                "training/dynamic_sampling/generated_prompt_groups": generated_groups,
                "training/dynamic_sampling/generated_rollouts": (
                    generated_groups * self.rollout_group_size
                ),
                "training/dynamic_sampling/generated_logical_rollouts": (
                    generated_groups * self.rollout_group_size
                ),
                "training/dynamic_sampling/valid_groups": valid_groups,
                "training/dynamic_sampling/keep_rate": valid_groups / generated_groups,
                "training/dynamic_sampling/all_zero_groups": int(
                    filtered.get(0.0, filtered.get("0.0", 0))
                ),
                "training/dynamic_sampling/all_one_groups": int(
                    filtered.get(1.0, filtered.get("1.0", 0))
                ),
            }
        )
        if self.rollout_counts_fn is not None:
            counts = self.rollout_counts_fn()
            delta = {key: value - self.rollout_counts_start[key] for key, value in counts.items()}
            metrics.update({
                "training/dynamic_sampling/physical_rollout_attempts": sum(delta.values()),
                "training/dynamic_sampling/initial_rollout_attempts": delta["initial"],
                "training/dynamic_sampling/infrastructure_replacements": delta["infrastructure"],
                "training/dynamic_sampling/user_replacements": delta["user"],
            })

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(
        self, global_steps: int, partition_id: str, batch_size: int
    ) -> tuple[KVBatchMeta, dict]:
        if partition_id != "train":
            return super().sample(global_steps, partition_id, batch_size)

        generated_batches = 1  # The trainer submits the first logical batch.
        last_debug_time = time.time()
        eviction_metrics: dict = {}
        exhausted = set()
        reuse_members = self._isolates_failed_groups(partition_id)

        while True:
            self._sync_metadata_from_transfer_queue()
            if self._isolates_failed_groups(partition_id):
                new_failures = self.failure_keys[partition_id] - exhausted
                self._audit_exhausted_groups(partition_id, new_failures)
                exhausted.update(new_failures)
                eviction_metrics["training/dynamic_sampling/exhausted_groups"] = len(exhausted)
            eviction_reasons = self._terminal_eviction_reasons(
                global_steps, partition_id
            )
            evicted_uids, stale_count, _dapo_count, new_metrics = (
                self._evict_terminal_groups(
                    global_steps, partition_id, eviction_reasons
                )
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, new_metrics, stale_count)

            sampleable_uids = self._sampleable_terminal_keys(
                partition_id, eviction_reasons
            )
            inflight_count = len(self.pending_keys[partition_id]) + len(
                self.running_keys[partition_id]
            )

            # A logical batch is evaluated only after all its prompts terminate.
            # This makes the 8 x 3 accounting exact and avoids policy-version mix.
            at_cap = generated_batches >= self.max_num_gen_batches
            if inflight_count == 0 and (
                len(sampleable_uids) >= batch_size
                or (reuse_members and at_cap and sampleable_uids)
            ):
                self._check_fatal_rollout()
                if reuse_members and any(
                    self.prompt_global_steps[partition_id][uid] != global_steps for uid in sampleable_uids
                ):
                    raise ValueError("usable rollout groups span different policy versions")
                selected_uids, partition_snapshot, _ = self._select_prompt_uids(
                    partition_id, sampleable_uids, len(sampleable_uids) if reuse_members else batch_size
                )
                surplus_uids = sampleable_uids - set(selected_uids)
                if surplus_uids:
                    self._clear_groups(partition_id, surplus_uids)
                    key = "training/filter_groups/discarded_surplus_samples"
                    eviction_metrics[key] = eviction_metrics.get(key, 0) + len(
                        surplus_uids
                    )
                self._add_sampling_metrics(
                    eviction_metrics,
                    generated_batches,
                    len(sampleable_uids),
                )
                selected_set = set(selected_uids)
                if not any(
                    key.split("_")[0] in selected_set for key in partition_snapshot
                ):
                    raise RuntimeError(
                        "selected groups contain no materializable trajectories"
                    )
                batch = self._materialize_batch(
                    partition_id, selected_uids, partition_snapshot
                )
                if reuse_members:
                    context = {uid: self.terminal_groups[partition_id][uid] for uid in selected_uids}
                    batch.extra_info["procredit_terminal_groups"] = context
                    prefix = "training/dynamic_sampling/"
                    eviction_metrics.update({
                        prefix + "selected_groups": len(selected_uids),
                        prefix + "selected_real_rollouts": len(batch.keys),
                        prefix + "salvaged_groups": sum(bool(tag["failed_session_ids"]) for tag in context.values()),
                        prefix + "salvaged_rollouts": sum(
                            len(self.credit_cache[uid]["members"]) for uid, tag in context.items()
                            if tag["failed_session_ids"]
                        ),
                        prefix + "cap_remainder_used": int(at_cap and len(selected_uids) < batch_size),
                        prefix + "rollout_use_rate": len(batch.keys) / (
                            generated_batches * self.conceptual_gen_batch_size * self.rollout_group_size
                        ),
                    })
                return batch, eviction_metrics

            if inflight_count == 0:
                self._check_fatal_rollout()
                if generated_batches >= self.max_num_gen_batches:
                    generated_prompts = (
                        generated_batches * self.conceptual_gen_batch_size
                    )
                    eviction_metrics["training/dynamic_sampling/cap_reached"] = 1
                    self._add_sampling_metrics(
                        eviction_metrics,
                        generated_batches,
                        len(sampleable_uids),
                    )
                    self._clear_attempt(partition_id)
                    raise DynamicSamplingCapReached(
                        generated_prompt_groups=generated_prompts,
                        generated_trajectories=generated_prompts
                        * self.rollout_group_size,
                        metrics=eviction_metrics,
                    )
                assert self.refill_fn is not None
                self.refill_fn(self.conceptual_gen_batch_size)
                generated_batches += 1
                continue

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)


class CappedPPOTrainerSync(PPOTrainerSync):
    """V1 synchronous trainer that retries after a capped, skipped update."""

    conceptual_gen_batch_size = 8
    max_num_gen_batches = 3
    rollout_group_size = 8

    def _procredit_runtime(self):
        enabled = self.config.algorithm.get("procredit_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("trainer ProCredit mode flag must be boolean")
        runtime_path = os.environ.get("AGENTIC_RL_CONFIG")
        if runtime_path is None:
            if enabled:
                raise ValueError("ProCredit mode requires its runtime reward config")
            return None  # Preserve standalone legacy trainer use.
        project = load_runtime_config(runtime_path)
        if enabled != is_procredit(project):
            raise ValueError("trainer ProCredit flag differs from runtime reward mode")
        if not enabled:
            return None
        root = Path(self.config.trainer.default_local_dir).parent / "group_audits"
        return CreditConfig.from_project(project), root

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        runtime = self._procredit_runtime()
        if runtime is None:
            return super()._compute_advantage(batch, metrics)
        config, audit_dir = runtime
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=["response_mask", "extra_fields"],
        )
        fields, credit_metrics = build_credit_tensors(
            batch.keys, list(data["extra_fields"]), data["response_mask"], config,
            audit_dir=audit_dir,
            terminal_groups=getattr(batch, "extra_info", {}).get("procredit_terminal_groups"),
            padding=[tag.get("is_padding", False) for tag in batch.tags] if getattr(batch, "tags", None) else None,
        )
        metrics.update(credit_metrics)
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id,
            fields=TensorDict(fields, batch_size=len(batch.keys)),
        )
        # Match the pinned parent: field writes must not replace metadata from
        # balancing (padding flags, temperature and terminal membership).
        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        self.replay_buffer._check_fatal_rollout()
        if not self.config.trainer.get("ppo_audit", False):
            return super()._update_actor(batch, metrics)
        from verl.workers.utils.padding import response_from_nested

        def rows(tensor):
            return [item.detach().float().cpu().tolist() for item in tensor.unbind()]

        def read_old():
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["old_log_probs", "response_mask"],
            )
            return rows(data["old_log_probs"]), rows(data["response_mask"])

        def compute_current():
            saved = dict(batch.extra_info)
            try:
                batch.extra_info.update(
                    calculate_entropy=False,
                    compute_loss=False,
                    temperature=self.config.actor_rollout_ref.rollout.temperature,
                )
                output = self.actor_rollout_wg.compute_log_prob(batch)
                if len(output) != len(batch):
                    raise ValueError("audit inference returned a different batch size")
                data = tq.kv_batch_get(
                    keys=batch.keys,
                    partition_id=batch.partition_id,
                    select_fields=["log_probs", "response_mask"],
                )
                return rows(
                    response_from_nested(data["log_probs"], data["response_mask"])
                )
            finally:
                batch.extra_info.clear()
                batch.extra_info.update(saved)

        parent_update = super()._update_actor
        result, report = audit_update(
            read_old=read_old,
            compute_current=compute_current,
            update=lambda: parent_update(batch, metrics),
            path=Path(self.config.trainer.default_local_dir).parent
            / "ppo_audit"
            / f"step_{self.global_steps}.json",
            tolerance=float(
                self.config.trainer.get("ppo_audit_logprob_tolerance", 0.005)
            ),
        )
        metrics.update(
            {
                f"audit/{key}": value
                for key, value in report.items()
                if key.startswith("ratio_")
            }
        )
        return result

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        checkpoint = (
            Path(self.config.trainer.default_local_dir)
            / f"global_step_{self.global_steps}"
        )
        self._write_step_counters(self.step_clock, checkpoint / "step_counters.json")
        snapshot_resume_identity(checkpoint.parent.parent, checkpoint)

    def _write_step_counters(
        self, clock: TrainingStepClock, path: Path | None = None
    ) -> None:
        path = (
            path
            or Path(self.config.trainer.default_local_dir).parent / "step_counters.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".step-counters.", suffix=".tmp", dir=path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "attempt_step": clock.attempt_step,
                        "optimizer_step": clock.optimizer_step,
                        "consecutive_skips": clock.consecutive_skips,
                    },
                    handle,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _save_last_valid_checkpoint(self, optimizer_step: int) -> None:
        """Save the unchanged latest model under its true optimizer step."""
        current_step = self.global_steps
        try:
            self.global_steps = optimizer_step
            self._save_checkpoint()
        finally:
            self.global_steps = current_step

    def _build_replay_buffer(self) -> CappedDynamicReplayBuffer:
        sampler = self.config.trainer.v1.sampler
        filter_groups = self.config.algorithm.filter_groups
        if not filter_groups.enable or filter_groups.metric != "train_reward":
            raise ValueError(
                "capped trainer requires filter_groups.metric=train_reward"
            )
        configured_cap = int(filter_groups.max_num_gen_batches)
        if configured_cap != self.max_num_gen_batches:
            raise ValueError(
                f"expected max_num_gen_batches={self.max_num_gen_batches}, got {configured_cap}"
            )
        if (
            int(self.config.data.train_batch_size) != 4
            or int(self.config.actor_rollout_ref.rollout.n) != 8
        ):
            raise ValueError(
                "capped trainer is fixed to 4 prompt groups x 8 trajectories"
            )
        credit_runtime = self._procredit_runtime()
        budget = api_budget()
        rollout_counts_fn = None
        if credit_runtime and credit_runtime[0].version == "procredit-turn-v4":
            def rollout_counts_fn():
                return budget.call("rollout_counts", "train")
        return CappedDynamicReplayBuffer(
            trainer_mode="sync",
            trainer_config=self.config.trainer.v1.sync,
            max_off_policy_threshold=sampler.max_off_policy_threshold,
            max_off_policy_strategy=sampler.max_off_policy_strategy,
            sampler_kwargs=sampler.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
            filter_groups_metric="train_reward",
            train_batch_size=4,
            # The parent validates this bookkeeping field when failed-group
            # refilling is enabled. Logical refills below are still eight.
            gen_batch_size=1,
            max_inflight_gen_batches=1,
            sync_refill_failed_groups=True,
            conceptual_gen_batch_size=self.conceptual_gen_batch_size,
            max_num_gen_batches=self.max_num_gen_batches,
            rollout_group_size=self.rollout_group_size,
            credit_config=credit_runtime[0] if credit_runtime else None,
            group_audit_dir=credit_runtime[1] if credit_runtime else None,
            rollout_counts_fn=rollout_counts_fn,
            fatal_error_fn=lambda: budget.call("fatal_rollout"),
        )

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        if self.config.trainer.critic_warmup != 0:
            raise ValueError("GRPO requires trainer.critic_warmup=0")
        if self.parameter_sync_step != 1:
            raise ValueError("capped trainer requires parameter_sync_step=1")
        self.replay_buffer.begin_attempt()
        self._add_prompts_to_generate(self.conceptual_gen_batch_size)
        self.local_trigger_step = 0
        return self._step_once(metrics, timing_raw, sample_batch_size=4)

    def fit(self, agent_loop_manager) -> None:
        """Train while treating a capped candidate batch as one skipped step."""
        if self.config.trainer.critic_warmup != 0:
            raise ValueError("GRPO requires trainer.critic_warmup=0")
        self.agent_loop_manager = agent_loop_manager
        SkipManager.init(self.config)
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            val_metrics = self._validate()
            self.on_validate_end()
            if not val_metrics:
                raise RuntimeError("initial validation returned no metrics")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // self.steps_per_epoch
        clock = TrainingStepClock(
            attempt_step=self.global_steps,
            optimizer_step=self.global_steps,
        )
        if self.config.trainer.resume_mode == "resume_path":
            clock = restore_step_clock(
                Path(self.config.trainer.resume_from_path), self.global_steps
            )
        self.step_clock = clock
        max_attempts_without_update = int(
            self.config.trainer.get("max_attempts_without_update", 50)
        )
        progress = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )
        self.global_steps += 1
        SkipManager.set_step(self.global_steps)
        self._reissue_inflight_prompts()
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False
        self.on_train_begin()
        last_val_metrics = None

        while (
            current_epoch < self.config.trainer.total_epochs
            and self.global_steps <= self.total_training_steps
        ):
            is_last_step = self.global_steps >= self.total_training_steps
            metrics: dict = {}
            self.timing_raw = {}
            batch: KVBatchMeta | None = None
            skipped = False

            with marked_timer("step", self.timing_raw):
                self.on_step_begin()
                self._start_profiling()
                try:
                    batch = self.step(metrics, self.timing_raw)
                except DynamicSamplingCapReached as exc:
                    skipped = True
                    metrics.update(exc.metrics)
                    metrics["training/dynamic_sampling/skipped_optimizer_steps"] = 1
                    # _step_once did not reach the normal on_sample_end hook.
                    self.on_sample_end()
                    logger.warning("%s; optimizer update skipped", exc)
                finally:
                    self._stop_profiling()

                clock.record_attempt(updated=not skipped)
                metrics["training/steps/attempt_step"] = clock.attempt_step
                metrics["training/steps/optimizer_step"] = clock.optimizer_step
                metrics["training/steps/consecutive_skips"] = clock.consecutive_skips
                self._write_step_counters(clock)

                if (
                    not skipped
                    and self.config.trainer.save_freq > 0
                    and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    )
                ):
                    with marked_timer(
                        "save_checkpoint", self.timing_raw, color="green"
                    ):
                        self._save_checkpoint()

                if skipped:
                    # Wake replicas without falsely stamping a new policy version.
                    self.checkpoint_manager.update_weights(clock.optimizer_step)
                else:
                    self.on_step_end()
                metrics.update(self._consume_sync_metrics())

            if (
                not skipped
                and self.config.trainer.test_freq > 0
                and (
                    is_last_step
                    or self.global_steps % self.config.trainer.test_freq == 0
                )
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    val_metrics = self._validate()
                    self.on_validate_end()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if batch is not None:
                self._compute_metrics(
                    batch,
                    metrics,
                    self.timing_raw,
                    global_steps=self.global_steps,
                    epoch=current_epoch,
                )
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(
                        batch,
                        self.timing_raw,
                        rollout_data_dir,
                    )
                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            else:
                metrics.update(
                    {
                        f"timing_s/{name}": value
                        for name, value in self.timing_raw.items()
                        if isinstance(value, int | float)
                    }
                )

            filtered_counts = metrics.pop(DAPO_FILTERED_REWARD_COUNTS_KEY, None)
            self.logger.log(data=metrics, step=clock.attempt_step)
            if filtered_counts:
                self.dapo_filtered_reward_logger.log(
                    self.config.trainer.logger,
                    filtered_counts,
                    clock.attempt_step,
                )

            if skipped:
                if clock.consecutive_skips >= max_attempts_without_update:
                    if clock.optimizer_step > 0:
                        self._save_last_valid_checkpoint(clock.optimizer_step)
                    self.on_train_end()
                    self._shutdown_dump_executor()
                    progress.close()
                    raise RuntimeError(
                        "dynamic sampling produced no optimizer batch for "
                        f"{clock.consecutive_skips} consecutive attempts"
                    )
                continue

            if clock.optimizer_step != self.global_steps:
                raise RuntimeError("optimizer-step accounting diverged from veRL")
            progress.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                self.on_train_end()
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress.close()
                return

        self.on_train_end()
        self._shutdown_dump_executor()
