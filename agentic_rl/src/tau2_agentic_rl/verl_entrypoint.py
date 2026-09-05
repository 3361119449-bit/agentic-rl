"""Hydra entry point that installs the bounded Tau2 trainer in veRL v0.9.0."""

from __future__ import annotations

import os

import hydra
import ray
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

from tau2_agentic_rl.concurrency import SharedBudget, limits_from_project
from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.verl_capped_trainer import CappedPPOTrainerSync


class Tau2TaskRunner:
    """Ray-side runner that constructs the project trainer explicitly."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        # Pinned veRL TaskRunnerV1 is already an ActorClass, not a base class.
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        manager_cls = (
            load_class_from_fqn(manager_fqn, "AgentLoopManager")
            if manager_fqn
            else AgentLoopManagerTQ
        )
        self.agent_loop_manager = manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config):
        import transfer_queue as tq

        configure_verl_logging()
        config.transfer_queue.enable = True
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            # Runner ownership keeps worker failures from resetting the budget.
            self.shared_budget = SharedBudget(
                limits_from_project(
                    load_runtime_config(os.environ["AGENTIC_RL_CONFIG"])
                ),
                require_ray=True,
            )
            if config.trainer.val_only:
                from verl.trainer.ppo.v1 import PPOTrainerSync

                self.trainer = PPOTrainerSync(config=config)
            else:
                self.trainer = CappedPPOTrainerSync(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(
    config_path="pkg://verl.trainer.config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config) -> None:
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(Tau2TaskRunner))


if __name__ == "__main__":
    main()
