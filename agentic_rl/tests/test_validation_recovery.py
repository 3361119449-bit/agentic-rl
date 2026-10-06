import time
import uuid
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_procredit_runtime import production_method
from test_sampling_isolation import make_buffer


@pytest.mark.parametrize("empty", [False, True])
def test_validation_consumes_successful_siblings_once_without_group_refill(scratch_dir, empty):
    buffer = make_buffer(scratch_dir)
    buffer.validation_group_size = 2
    buffer.finished_keys["val"] = set() if empty else {"good"}
    buffer.failure_keys["val"] = {"bad"}
    buffer.pending_keys["val"], buffer.running_keys["val"] = set(), set()
    buffer.partitions["val"] = {"bad_0_0": {}, "bad_1_0": {}}
    if not empty:
        buffer.partitions["val"].update({"good_0_0": {}, "good_1_0": {}, "good_1_1": {}})
    # Both failed sessions wrote an output before postprocessing failed.
    tags = {"val": {
        "bad": {"status": "failure", "rollout_n": 2, "failed_session_ids": [0, 1],
                "failure_kind": "transient_exhausted"},
        "good": {"status": "finished", "rollout_n": 2, "failed_session_ids": [],
                 "failure_kind": None},
    }, "train": buffer.terminal_groups["train"]}
    buffer.terminal_metadata_fn = lambda: tags
    batch, metrics = buffer.sample(1, "val", 1 if empty else 2)
    assert batch.keys == ([] if empty else ["good_0_0", "good_1_0", "good_1_1"])
    assert metrics["val/coverage/requested_slots"] == (2 if empty else 4)
    assert metrics["val/coverage/completed_slots"] == (0 if empty else 2)
    assert metrics["val/coverage/failed_slots"] == 2


def validation_trainer(results):
    class Input(dict):
        def __len__(self):
            return len(self["raw_prompt"])

    class Batch(list):
        def __init__(self, keys):
            super().__init__(keys)
            self.keys, self.partition_id = keys, "val"

    class Text:
        def to_padded_tensor(self, **kwargs):
            return np.array([[1]])

    def get(**kwargs):
        if kwargs["select_fields"] == ["prompts", "responses"]:
            return {"prompts": Text(), "responses": Text()}
        return {
            "uid": np.array(["good"]), "num_turns": np.array([1]),
            "rm_scores": SimpleNamespace(sum=lambda dim: np.array([1.0])),
            "data_source": np.array(["airline"]),
        }

    cls = production_method("CappedPPOTrainerSync", "_validate", object,
        Counter=Counter, defaultdict=defaultdict, np=np, uuid=uuid,
        tu=SimpleNamespace(get_tensordict=Input, assign_non_tensor_data=lambda *a: None),
        tq=SimpleNamespace(kv_batch_put=lambda **kw: None, kv_batch_get=get, kv_clear=lambda **kw: None),
        time=time,
    )
    trainer = cls()
    trainer.global_steps = 1
    trainer.val_dataloader = [{"raw_prompt": ["p"]} for _ in results]
    batches = iter([
        (Batch(keys), {"val/coverage/requested_slots": 2,
                       "val/coverage/completed_slots": int(bool(keys)),
                       "val/coverage/failed_slots": 2 - int(bool(keys))})
        for keys in results
    ])
    trainer.replay_buffer = SimpleNamespace(sample=lambda **kw: next(batches))
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda b: None)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *a, **kw: "text")
    trainer.config = SimpleNamespace(trainer={})
    trainer._maybe_log_val_generations = lambda **kw: None
    seen = []

    def metrics(sources, uids, rewards, turns):
        seen.append(uids)
        assert uids and len(uids) == len(rewards["reward"]) == len(turns)
        return {"val/completed_reward": 1.0}

    trainer._val_metrics_update = metrics
    return trainer, seen


def test_empty_validation_returns_coverage_without_reward_or_update_failure():
    trainer, seen = validation_trainer([[], []])
    metrics = trainer._validate()
    assert not seen
    assert metrics["val/coverage/requested_slots"] == 4
    assert metrics["val/coverage/completed_slots"] == 0
    assert metrics["val/coverage/completion_rate"] == 0
    assert "val/completed_reward" not in metrics


def test_partial_validation_counts_missing_slots_without_fake_rewards():
    trainer, seen = validation_trainer([[], ["good_0_0"]])
    metrics = trainer._validate()
    assert seen == [["good"]]
    assert metrics["val/coverage/requested_slots"] == 4
    assert metrics["val/coverage/completed_slots"] == 1
    assert metrics["val/coverage/completion_rate"] == 0.25
    assert metrics["val/completed_reward"] == 1
