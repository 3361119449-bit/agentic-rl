"""CPU checks of the actual nested TensorDict sent to the veRL actor."""

import pytest

from scripts.saved_rollout_support.batch import SavedBatch, SavedTrajectory

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")


def batch_fixture():
    row = SavedTrajectory(
        "a", [10, 11], [12, 90, 13], [1, 0, 1], [-0.1, 0, -0.2], [0.5, 0, -0.5], 2
    )
    return SavedBatch([row], {}, {}, {"padding_rows": 1})


def test_tensor_batch_retains_causal_positions_and_zero_loss_padding():
    from scripts.saved_rollout_support.runtime import build_tensor_batch

    data = build_tensor_batch(batch_fixture(), eos_token_id=2)
    assert list(data.batch_size) == [2]
    assert data["input_ids"].unbind()[0].tolist() == [10, 11, 12, 90, 13]
    assert data["position_ids"].unbind()[0].tolist() == [0, 1, 2, 3, 4]
    assert data["response_mask"].unbind()[0].tolist() == [1, 0, 1]
    assert data["response_mask"].unbind()[1].tolist() == [0]
    assert data["advantages"].unbind()[0].tolist() == [0.5, 0, -0.5]
    assert data["old_log_probs"].unbind()[0].tolist() == pytest.approx([-0.1, 0, -0.2])
    assert data["old_log_probs"].dtype == torch.float32
    assert data["loss_mask"].values().tolist() == [1, 0, 1, 0]


def test_worker_snapshot_detects_changed_old_probs_advantages_or_mask():
    from scripts.saved_rollout_support.runtime import (
        build_tensor_batch,
        frozen_batch_digest,
    )

    data = build_tensor_batch(batch_fixture(), eos_token_id=2)
    before = frozen_batch_digest(data)
    data["old_log_probs"].values()[0] += 0.1
    assert before != frozen_batch_digest(data)
    data = build_tensor_batch(batch_fixture(), eos_token_id=2)
    before = frozen_batch_digest(data)
    data["advantages"].values()[0] += 1
    assert before != frozen_batch_digest(data)


def test_parameter_digest_tracks_trainable_weights_only():
    from scripts.saved_rollout_support.runtime import trainable_parameter_digest

    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    model.bias.requires_grad_(False)
    before = trainable_parameter_digest(model)
    with torch.no_grad():
        model.bias.add_(1)
    assert before == trainable_parameter_digest(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert before != trainable_parameter_digest(model)
