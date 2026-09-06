"""Run only with real pinned veRL/Ray/vLLM installed; no mocks or model weights."""

import asyncio
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import verl
from hydra import compose, initialize_config_module
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

from scripts.evaluate_airline import build_evaluation_command
from scripts.train_airline_grpo import VERL_COMMIT, build_command
from tau2_agentic_rl.concurrency import SharedBudget
from tau2_agentic_rl.config import load_yaml

ROOT = Path(__file__).parents[1]


def test_real_runner_import_and_single_ray_decoration():
    checkout = Path(verl.__file__).resolve().parents[1]
    sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == VERL_COMMIT
    module = importlib.import_module("tau2_agentic_rl.verl_entrypoint")
    assert isinstance(module.Tau2TaskRunner, type)
    runner = module.Tau2TaskRunner()
    assert runner.config is runner.trainer is runner.agent_loop_manager is None
    assert ray.remote(num_cpus=1)(module.Tau2TaskRunner) is not None


@pytest.mark.parametrize("mode", ["training", "baseline_eval", "adapter_eval"])
def test_actual_launcher_config_passes_real_verl_validation(mode, tmp_path):
    if mode == "training":
        command = build_command(
            project_root=ROOT,
            model_path="/unused-model",
            train_file=tmp_path / "train.parquet",
            val_file=tmp_path / "val.parquet",
            total_epochs=1,
            extra=[],
        )
    else:
        adapter = None
        if mode == "adapter_eval":
            adapter = tmp_path / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 32, "lora_alpha": 64})
            )
        args = SimpleNamespace(
            model_path="/unused-model",
            seed=42,
            tag="contract",
            lora_adapter=adapter,
            extra=[],
        )
        project = load_yaml(ROOT / "configs/evaluation/airline_eval_v1.yaml")
        command = build_evaluation_command(
            args,
            project_root=ROOT,
            data_file=tmp_path / "eval.parquet",
            run_root=tmp_path,
            project=project,
            identity=project["rollout"],
        )
    with initialize_config_module(
        config_module="verl.trainer.config", version_base=None
    ):
        config = compose(config_name="ppo_trainer", overrides=command[3:])
    validate_config(config, need_reference_policy(config), need_critic(config))
    if mode != "training":
        assert config.trainer.val_only is True


def test_two_real_ray_workers_share_one_trajectory_and_api_budget():
    ray.init(num_cpus=2, include_dashboard=False)
    try:
        limits = {"trajectories": 2, "user_api": 1, "judge_api": 1}
        budget = SharedBudget(limits, require_ray=True)  # driver ownership

        @ray.remote(num_cpus=0)
        class Worker:
            async def run(self):
                shared = SharedBudget(limits, require_ray=True)

                async def trajectory():
                    async with shared.aslot("trajectories") as lease:
                        async with shared.aslot("user_api"):
                            await asyncio.sleep(0.02)
                        async with shared.aslot("judge_api"):
                            await asyncio.sleep(0.02)
                        lease.progress()

                await asyncio.gather(*(trajectory() for _ in range(8)))

        workers = [Worker.remote(), Worker.remote()]
        ray.get([worker.run.remote() for worker in workers], timeout=120)
        metrics = budget.call("snapshot")
        assert metrics["peak_active_trajectories"] == 2
        assert (
            metrics["peak_user_api_inflight"] == metrics["peak_judge_api_inflight"] == 1
        )
        assert set(metrics["active"].values()) == {0}
        assert metrics["progress_revision"]["trajectories"] == 16 * 3
    finally:
        ray.shutdown()


def test_actual_agentloop_helper_truncates_but_project_renderer_preserves_tokens():
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase

    from tau2_agentic_rl.agent_loop.airline import Tau2AirlineAgentLoop

    async def check():
        instance = Tau2AirlineAgentLoop.__new__(Tau2AirlineAgentLoop)
        instance.loop = asyncio.get_running_loop()
        instance.rollout_config = SimpleNamespace(prompt_length=8192)
        instance.processor = None
        instance.apply_chat_template_kwargs = {}
        instance.system_prompt = []
        tokens = list(range(9000))
        instance.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *a, **kw: tokens
        )
        messages = [{"role": "system", "content": "fixture policy"}]
        assert (
            await AgentLoopBase.apply_chat_template(instance, messages)
            == tokens[-8192:]
        )
        assert await instance._render_full_chat(messages) == tokens

    asyncio.run(check())
