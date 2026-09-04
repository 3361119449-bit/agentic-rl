"""Hydra entry point that installs the bounded Tau2 trainer in veRL v0.9.0."""

from __future__ import annotations

import hydra
import ray
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging

from tau2_agentic_rl.verl_capped_trainer import CappedPPOTrainerSync


class Tau2TaskRunner(TaskRunnerV1):
    """Ray-side runner that constructs the project trainer explicitly."""

    def run(self, config):
        import transfer_queue as tq

        configure_verl_logging()
        config.transfer_queue.enable = True
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
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
