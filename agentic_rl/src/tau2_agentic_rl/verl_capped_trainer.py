"""Pinned veRL v0.9.0 trainer extensions for bounded dynamic sampling.

veRL's V1 replay buffer intentionally ignores ``max_num_gen_batches``.  The
experiment plan, however, requires exactly three logical generation batches of
eight prompts and no optimizer update when fewer than four mixed-reward groups
remain.  This module supplies that narrowly scoped behavior.  It deliberately
uses veRL V1 protected APIs and must only be used with the commit checked by the
launcher.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pprint import pprint

import transfer_queue as tq
from omegaconf import OmegaConf
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
            f"{self.generated_trajectories} trajectories"
        )


class CappedDynamicReplayBuffer(ReplayBuffer):
    """Synchronous group filter with a hard logical-generation-batch cap."""

    def __init__(
        self,
        *args,
        conceptual_gen_batch_size: int,
        max_num_gen_batches: int,
        rollout_group_size: int,
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

    def _clear_attempt(self, partition_id: str) -> None:
        """Remove every prompt and trajectory left by the capped attempt."""
        self._sync_metadata_from_transfer_queue()
        all_prompt_uids = set(self.prompt_global_steps[partition_id])
        self._clear_groups(partition_id, all_prompt_uids)

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

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(
        self, global_steps: int, partition_id: str, batch_size: int
    ) -> tuple[KVBatchMeta, dict]:
        if partition_id != "train":
            return super().sample(global_steps, partition_id, batch_size)

        generated_batches = 1  # The trainer submits the first logical batch.
        last_debug_time = time.time()
        eviction_metrics: dict = {}

        while True:
            self._sync_metadata_from_transfer_queue()
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
            if inflight_count == 0 and len(sampleable_uids) >= batch_size:
                selected_uids, partition_snapshot, _ = self._select_prompt_uids(
                    partition_id, sampleable_uids, batch_size
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
                return self._materialize_batch(
                    partition_id, selected_uids, partition_snapshot
                ), eviction_metrics

            if inflight_count == 0:
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
        )

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        if self.parameter_sync_step != 1:
            raise ValueError("capped trainer requires parameter_sync_step=1")
        self._add_prompts_to_generate(self.conceptual_gen_batch_size)
        self.local_trigger_step = 0
        return self._step_once(metrics, timing_raw, sample_batch_size=4)

    def fit(self, agent_loop_manager) -> None:
        """Train while treating a capped candidate batch as one skipped step."""
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

                self.on_step_end()
                metrics.update(self._consume_sync_metrics())

            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
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
            self.logger.log(data=metrics, step=self.global_steps)
            if filtered_counts:
                self.dapo_filtered_reward_logger.log(
                    self.config.trainer.logger,
                    filtered_counts,
                    self.global_steps,
                )

            progress.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress.close()
                return

        self.on_train_end()
        self._shutdown_dump_executor()
