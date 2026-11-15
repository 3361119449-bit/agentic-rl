"""Exercise the launcher's LoRA export on real single-GPU FSDP, without APIs."""

from pathlib import Path

import pytest

from scripts.train_airline_grpo import FULL_TRAIN_CONFIG, build_command
from tau2_agentic_rl.config import load_yaml


def test_single_gpu_launcher_exports_complete_lora_weights(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA device and pinned veRL")
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from verl.utils.fsdp_utils import collect_lora_params

    root = Path(__file__).parents[1]
    command = build_command(
        project_root=root,
        model_path="/unused-model",
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        total_epochs=1,
        extra=[],
        project_config=load_yaml(root / FULL_TRAIN_CONFIG),
    )
    overrides = dict(item.split("=", 1) for item in command[3:])
    assert overrides["trainer.n_gpus_per_node"] == "1"
    assert overrides["actor_rollout_ref.rollout.load_format"] == "safetensors"
    layered = overrides["actor_rollout_ref.rollout.layered_summon"] == "true"

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1,
    )
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 16, bias=False),
            torch.nn.Linear(16, 16, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)
        peft_model = get_peft_model(
            model, LoraConfig(r=2, lora_alpha=4, target_modules=["0", "1"]),
        ).to(dtype=torch.bfloat16)
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.125)
        expected = {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(peft_model).items()
        }
        base_before = {
            name: value.detach().cpu().clone()
            for name, value in peft_model.named_parameters() if "lora_" not in name
        }
        wrapped = FSDP(peft_model, use_orig_params=True, device_id=0)
        assert wrapped.sharding_strategy == ShardingStrategy.NO_SHARD
        actual = collect_lora_params(wrapped, layered_summon=layered, base_sync_done=True)
        assert actual.keys() == expected.keys()
        assert len(actual) == 4
        for name, value in actual.items():
            assert value.device.type == "cpu"
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        for name, value in peft_model.named_parameters():
            if name in base_before:
                torch.testing.assert_close(value.cpu(), base_before[name], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()
