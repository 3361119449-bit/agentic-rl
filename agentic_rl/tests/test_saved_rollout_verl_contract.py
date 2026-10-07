"""Optional checks against the project's real, pinned veRL checkout."""

import os
from pathlib import Path

import pytest

ROOT = os.environ.get("VERL_REPLAY_TEST_ROOT")
RESULTS = os.environ.get("SAVED_ROLLOUT_TEST_RESULTS")


@pytest.mark.skipif(
    not ROOT or not RESULTS, reason="requires pinned veRL checkout and saved results"
)
def test_compose_saved_actor_configuration_without_rollout_initialization(scratch_dir):
    pytest.importorskip("hydra")
    from scripts.saved_rollout_support.batch import load_saved_batch
    from scripts.saved_rollout_support.runtime import compose_actor_config

    batch = load_saved_batch(Path(RESULTS))
    config = compose_actor_config(
        batch, Path("original-model"), scratch_dir / "output", Path(ROOT)
    )
    actor = config.actor_rollout_ref.actor
    assert actor.ppo_epochs == 2
    assert actor.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n == 32
    assert actor.optim.lr == 5e-6
    assert actor.optim.total_training_steps == 1
    assert config.actor_rollout_ref.model.lora_rank == 32
    assert actor.fsdp_config.model_dtype == "bfloat16"
    assert actor.fsdp_config.mixed_precision.reduce_dtype == "float32"
    assert actor.policy_loss.loss_mode == "vanilla"
    assert config.algorithm.rollout_correction.bypass_mode is True


@pytest.mark.skipif(not ROOT, reason="requires pinned veRL checkout")
def test_saved_tensor_fields_work_with_native_verl_response_slicing():
    import sys

    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    pytest.importorskip("ray")
    sys.path.insert(0, ROOT)
    from verl.workers.utils.padding import no_padding_2_padding, response_from_nested

    from scripts.saved_rollout_support.batch import SavedBatch, SavedTrajectory
    from scripts.saved_rollout_support.runtime import build_tensor_batch

    row = SavedTrajectory(
        "a", [10, 11], [12, 90, 13], [1, 0, 1], [-0.1, 0, -0.2], [0.5, 0, -0.5], 0
    )
    data = build_tensor_batch(
        SavedBatch([row], {}, {}, {"padding_rows": 1}), eos_token_id=2
    )
    output = torch.nested.as_nested_tensor(
        [torch.tensor([9.0, 1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0])],
        layout=torch.jagged,
    )
    response = response_from_nested(output, data["response_mask"])
    assert response.unbind()[0].tolist() == [1, 2, 3]
    assert no_padding_2_padding(output, data).tolist() == [[1, 2, 3], [5, 0, 0]]


@pytest.mark.skipif(not ROOT, reason="requires pinned veRL checkout")
def test_native_ppo_updates_policy_tokens_only_and_keeps_old_probs_for_two_epochs():
    import sys

    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    pytest.importorskip("ray")
    sys.path.insert(0, ROOT)
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.config import FSDPActorConfig

    from scripts.saved_rollout_support.batch import SavedBatch, SavedTrajectory
    from scripts.saved_rollout_support.runtime import (
        build_tensor_batch,
        frozen_batch_digest,
    )

    row = SavedTrajectory(
        "a", [10, 11], [12, 90, 13], [1, 0, 1], [-0.1, 0, -0.2], [0.5, 0, -0.5], 0
    )
    data = build_tensor_batch(
        SavedBatch([row], {}, {}, {"padding_rows": 1}), eos_token_id=2
    )
    before = frozen_batch_digest(data)
    values = data.select(
        "old_log_probs", "advantages", "response_mask"
    ).to_padded_tensor()
    current = torch.nn.Parameter(torch.tensor([[-0.1, 1.5, -0.2], [0.0, 0.0, 0.0]]))
    original = current.detach().clone()
    optimizer = torch.optim.SGD([current], lr=0.01)
    config = FSDPActorConfig(
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=10.0,
    )
    for _ in range(2):
        optimizer.zero_grad()
        loss, _ = compute_policy_loss_vanilla(
            values["old_log_probs"],
            current,
            values["advantages"],
            values["response_mask"].to(bool),
            config=config,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert current.grad[0, 1].item() == 0
        assert current.grad[1].tolist() == [0, 0, 0]
        optimizer.step()
    assert current[0, 0].item() > original[0, 0].item()
    assert current[0, 2].item() < original[0, 2].item()
    assert frozen_batch_digest(data) == before
