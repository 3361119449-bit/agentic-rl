"""Single-rank CPU contract with the pinned, Muon-patched veRL training engine."""

import io
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_module
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def verify_fsdp_muon(overrides):
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        actor = compose(config_name="ppo_trainer", overrides=overrides).actor_rollout_ref.actor
    assert actor.fsdp_config.use_orig_params is False
    policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    def wrap(module):
        return FSDP(module, use_orig_params=actor.fsdp_config.use_orig_params, device_id=torch.device("cpu"),
                    sharding_strategy=ShardingStrategy.NO_SHARD, mixed_precision=policy)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(96, 64, bias=False, dtype=torch.bfloat16).requires_grad_(False)
            self.lora_A = wrap(nn.Linear(96, 32, bias=False))
            b = nn.Linear(32, 64, bias=False)
            nn.init.zeros_(b.weight)
            self.lora_B = wrap(b)

        def forward(self, x):
            return self.base(x) + 2 * self.lora_B(self.lora_A(x))

    def make_engine():
        engine = SimpleNamespace(module=wrap(Adapter()), engine_config=actor.fsdp_config,
                                 optimizer_config=omega_conf_to_dataclass(actor.optim),
                                 _fp32_lora_master=True, _fp32_optimizer_verified=False, _qat_enabled=False)
        engine.optimizer = FSDPEngine._build_optimizer(engine, engine.module)
        assert type(engine.optimizer) is torch.optim.Muon
        group, = engine.optimizer.param_groups
        assert (group["lr"], group["weight_decay"], group["adjust_lr_fn"]) == (1e-5, .01, "match_rms_adamw")
        assert (group["momentum"], group["nesterov"], group["ns_steps"], group["eps"]) == (.95, True, 5, 1e-7)
        assert tuple(group["ns_coefficients"]) == (3.4445, -4.775, 2.0315)
        assert all(p.ndim == 1 for p in group["params"])
        assert sorted(tuple(shape) for _, shape in engine._tau2_muon_matrices) == [(32, 96), (64, 32)]
        return engine

    def step(engine, x):
        engine.optimizer.zero_grad()
        output = engine.module(x)
        assert output.dtype == torch.bfloat16
        output.float().square().mean().backward()
        assert all(p.grad.dtype == torch.float32 for p in engine.module.parameters() if p.requires_grad)
        assert FSDPEngine.optimizer_step(engine) > 0
        assert engine._fp32_optimizer_verified
        for p, state in engine.optimizer.state.items():
            assert state["momentum_buffer"].shape == p._shapes[0]
            assert p.ndim == p.grad.ndim == 1
            assert state["momentum_buffer"].dtype == p.dtype == torch.float32

    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        torch.manual_seed(42)
        engine = make_engine()
        x = torch.randn(8, 96)
        start = {k: v.clone() for k, v in engine.module.state_dict().items()}
        step(engine, x)
        checkpoint = io.BytesIO()
        torch.save((engine.module.state_dict(), engine.optimizer.state_dict()), checkpoint)
        step(engine, x)
        resumed = make_engine()
        checkpoint.seek(0)
        model_state, optim_state = torch.load(checkpoint, weights_only=True)
        resumed.module.load_state_dict(model_state)
        resumed.optimizer.load_state_dict(optim_state)
        step(resumed, x)
        for name, weight in engine.module.state_dict().items():
            assert torch.equal(resumed.module.state_dict()[name], weight)
            assert torch.equal(start[name], weight) == name.startswith("base.")
        for p, q in zip(engine.optimizer.state, resumed.optimizer.state, strict=True):
            assert torch.equal(engine.optimizer.state[p]["momentum_buffer"],
                               resumed.optimizer.state[q]["momentum_buffer"])
    finally:
        dist.destroy_process_group()


def test_muon_real_engine_and_checkpoint():
    from scripts.train_airline_grpo import build_command
    from tau2_agentic_rl.config import load_yaml
    from tau2_agentic_rl.training_config import effective_project_config

    root = Path(__file__).resolve().parents[1]
    project = load_yaml(root / "configs/rl/airline_procredit_v4_train30_c32_routed.yaml")
    command = build_command(project_root=root, model_path="/model", train_file=Path("/train"),
                            val_file=Path("/val"), total_epochs=1, extra=[],
                            project_config=effective_project_config(project, [], optimizer_name="muon"))
    verify_fsdp_muon(command[3:])
