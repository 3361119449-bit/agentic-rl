"""Small LoRA-only adapter around the pinned veRL SFT trainer.

Keep the upstream optimizer/loss/fit path. Fix seeded loaders, complete small
validation sets, and epoch-boundary checkpoint restoration here, not in veRL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agentic_rl/src"))

from tau2_agentic_rl.base_identity import save_base_identity
from tau2_agentic_rl.sft_contract import (
    COMPLETE_MANIFEST,
    RUN_MANIFEST,
    publish_checkpoint,
    validate_resume,
    validate_runtime,
)


class EpochCheckpointHandler:
    def __init__(self, inner, identity, steps_per_epoch):
        self.inner = inner
        self.identity = identity
        self.steps_per_epoch = steps_per_epoch

    def load_checkpoint(self):
        if self.inner.resume_mode == "disable":
            return 0
        path, step = validate_resume(self.inner.resume_from_path, self.identity)
        # Restores LoRA/model, optimizer, LR scheduler AND RNG. At an epoch
        # boundary the saved loader has yielded its last batch but may not yet
        # have raised StopIteration. Restoring that iterator would consume an
        # empty epoch in upstream fit(). Use the fresh, seeded next-epoch loader.
        # Its original data_N.pt files must still pass the completeness check.
        self.inner.engine.load_checkpoint(str(path))
        return step

    def save_checkpoint(self, step):
        import torch

        path = Path(self.inner.default_local_dir) / f"global_step_{step}"
        if step <= 0 or step % self.steps_per_epoch:
            raise ValueError(
                "LoRA SFT checkpoints must be at completed epoch boundaries"
            )
        if (path / COMPLETE_MANIFEST).exists():
            raise FileExistsError(f"Checkpoint already completed: {path}")
        # Upstream synchronizes after all model/optimizer/loader files are saved.
        self.inner.save_checkpoint(step)
        if self.inner.rank == 0:
            base = json.loads(
                (path.parent / "base_model_identity.json").read_text(encoding="utf-8")
            )
            save_base_identity(path, base)
            publish_checkpoint(path, self.identity, step)
        torch.distributed.barrier()


def create_trainer_class(upstream):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    class LoraSFTTrainer(upstream.SFTTrainer):
        def _build_dataloader(self):
            config = self.config
            if config.model.lora_rank <= 0:
                raise ValueError(
                    "This entry point supports LoRA SFT only (lora_rank > 0)"
                )
            dp_rank = self.engine.get_data_parallel_rank()
            dp_size = self.engine.get_data_parallel_size()
            self.global_batch_size = config.data.train_batch_size
            if (
                self.global_batch_size % dp_size
                or len(self.train_dataset) < self.global_batch_size
            ):
                raise ValueError(
                    "Training data must contain a full global batch, divisible across DP ranks"
                )
            self.train_batch_size_per_dp = self.global_batch_size // dp_size
            self.collate_fn = upstream.SFTTensorCollator(config.data.pad_mode)
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                num_replicas=dp_size,
                rank=dp_rank,
                drop_last=True,
                seed=config.trainer.seed,
            )
            self.train_dataloader = StatefulDataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.train_sampler,
                collate_fn=self.collate_fn,
                num_workers=config.data.num_workers,
                drop_last=True,
                generator=torch.Generator().manual_seed(config.trainer.seed),
            )
            self.val_dataloader = None
            if self.val_dataset is not None and len(self.val_dataset):
                # Replicate the complete small validation set on every DP rank.
                # All ranks make equal collective calls, including N < DP size;
                # no DistributedSampler padding/duplicates or dropped tail.
                # Batch=1 makes upstream's mean exactly the per-example NLL mean.
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=self.collate_fn,
                    num_workers=config.data.num_workers,
                    generator=torch.Generator().manual_seed(config.trainer.seed),
                )
            config.trainer.test_freq = (
                "after_each_epoch" if self.val_dataloader is not None else -1
            )
            config.trainer.save_freq = "after_each_epoch"

        def _build_ckpt_handler(self):
            super()._build_ckpt_handler()
            identity = json.loads(
                (Path(self.config.trainer.default_local_dir) / RUN_MANIFEST).read_text(
                    encoding="utf-8"
                )
            )
            self.ckpt_handler = EpochCheckpointHandler(
                self.ckpt_handler, identity, self.steps_per_epoch
            )

    return LoraSFTTrainer


def main():
    validate_runtime()
    import hydra
    from transformers import set_seed
    from verl.trainer import sft_trainer as upstream
    from verl.utils.distributed import initialize_global_process_group

    trainer_class = create_trainer_class(upstream)

    @hydra.main(
        config_path=str(Path(upstream.__file__).resolve().parent / "config"),
        config_name="sft_trainer_engine",
        version_base=None,
    )
    def launch(config):
        upstream.auto_set_device(config)
        # Seed every worker before constructing LoRA/optimizer. Loader generators
        # are independent, so validation cannot consume training's global RNG.
        set_seed(config.trainer.seed)
        initialize_global_process_group()
        try:
            trainer_class(config).fit()
        finally:
            upstream.destroy_global_process_group()

    launch()


if __name__ == "__main__":
    main()
