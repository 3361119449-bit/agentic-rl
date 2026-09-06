"""Pinned-source/real-dependency CPU checks. No model weights or CUDA job.

Set SFT_VERL_ROOT to the pinned checkout. RUN_REAL_SFT_CONTRACT=1 additionally
imports the entire installed veRL trainer; otherwise isolate its exact methods.
"""

import ast
import importlib.util
import os
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def runtime():
    path = Path(__file__).parents[2] / "training/qwen3_4b_sft/lora_sft_runtime.py"
    spec = importlib.util.spec_from_file_location("lora_runtime_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def upstream(monkeypatch):
    if os.getenv("RUN_REAL_SFT_CONTRACT") == "1":
        from verl.trainer import sft_trainer as mod
    else:
        root = Path(os.environ["SFT_VERL_ROOT"])
        source = root / "verl/trainer/sft_trainer.py"
        cls = next(
            n
            for n in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "SFTTrainer"
        )
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"fit", "_init_engine", "_get_batch_seqlens"}
        ]
        scope = {"torch": torch, "logger": None, "NonTensorData": lambda x: x}
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
            scope,
        )
        trainer = type(
            "PinnedSourceTrainer", (), {n.name: scope[n.name] for n in methods}
        )
        collator_source = root / "verl/utils/dataset/dataset_utils.py"
        collator_nodes = [
            n
            for n in ast.parse(collator_source.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef)
            and n.name in {"DatasetPadMode", "SFTTensorCollator"}
        ]
        collator_scope = {"torch": torch, "Enum": Enum}
        exec(
            compile(
                ast.Module(body=collator_nodes, type_ignores=[]),
                str(collator_source),
                "exec",
            ),
            collator_scope,
        )
        mod = NS(
            SFTTrainer=trainer,
            SFTTensorCollator=collator_scope["SFTTensorCollator"],
            _scope=scope,
        )

    def quiet(*a, **kw):
        return None

    replacements = {
        "log_with_rank": quiet,
        "aggressive_empty_cache": quiet,
        "log_gpu_memory_usage": quiet,
        "tqdm": lambda x, **kw: x,
        "tu": NS(
            get_tensordict=lambda **kw: kw["tensor_dict"], assign_non_tensor=quiet
        ),
    }
    for key, value in replacements.items():
        if hasattr(mod, "_scope"):
            mod._scope[key] = value
        else:
            monkeypatch.setattr(mod, key, value)
    monkeypatch.setattr(torch.distributed, "barrier", quiet)
    return mod


def examples(count):
    return [
        {
            "input_ids": torch.tensor([i, i + 1, 2]),
            "loss_mask": torch.tensor([0, 1, 1]),
            "position_ids": torch.arange(3),
        }
        for i in range(count)
    ]


def build_trainer(runtime, upstream, *, seed=42, val_rows=0, dp=1, rank=0):
    cls = runtime.create_trainer_class(upstream)
    trainer = cls.__new__(cls)
    trainer.config = NS(
        model=NS(lora_rank=64, use_remove_padding=True),
        trainer=NS(seed=seed, total_epochs=2, total_training_steps=None),
        data=NS(
            train_batch_size=4,
            pad_mode="no_padding",
            num_workers=0,
            use_dynamic_bsz=True,
            max_token_len_per_gpu=16384,
            micro_batch_size_per_gpu=1,
        ),
    )
    trainer.engine = NS(
        get_data_parallel_rank=lambda: rank,
        get_data_parallel_size=lambda: dp,
        get_data_parallel_group=lambda: None,
        is_mp_src_rank_with_outputs=lambda: False,
    )
    trainer.train_dataset, trainer.val_dataset = (
        examples(12),
        examples(val_rows) if val_rows else None,
    )
    trainer._build_dataloader()
    trainer.optimizer_config = NS()
    trainer.trained, trainer.validated, trainer.saved = [], [], []

    def record(destination, data):
        destination.extend(int(row[0]) for row in data["input_ids"].unbind())

    trainer.training_client = NS(
        reset=lambda: None,
        train_batch=lambda data: record(trainer.trained, data),
        infer_batch=lambda data: record(trainer.validated, data),
    )
    trainer._init_engine()
    trainer.resume_global_step = 0
    trainer.model_config = NS(tokenizer=NS(pad_token_id=0))
    trainer.rank, trainer.device_name = 0, "cpu"
    trainer.start_profile_step = trainer.end_profile_step = -1
    trainer.ckpt_handler = NS(save_checkpoint=lambda step: trainer.saved.append(step))
    return trainer


def test_no_validation_finishes_and_saves_both_epochs(runtime, upstream):
    trainer = build_trainer(runtime, upstream)
    assert trainer.test_freq == -1
    trainer.fit()
    assert len(trainer.trained) == 24
    assert trainer.saved == [3, 6]


@pytest.mark.parametrize("val_rows", [1, 3, 10])
def test_validation_includes_each_example_without_tail_loss(
    runtime, upstream, val_rows
):
    trainer = build_trainer(runtime, upstream, val_rows=val_rows)
    trainer.fit()
    assert trainer.validated == list(range(val_rows)) * 2
    assert trainer.saved == [3, 6]


def test_validation_smaller_than_dp_has_equal_collective_counts(runtime, upstream):
    orders = []
    for rank in range(4):
        trainer = build_trainer(runtime, upstream, val_rows=1, dp=4, rank=rank)
        orders.append([int(row["input_ids"][0][0]) for row in trainer.val_dataloader])
    assert orders == [[0]] * 4


def test_seed_controls_real_sampler_and_is_repeatable(runtime, upstream):
    first = build_trainer(runtime, upstream, seed=42)
    repeat = build_trainer(runtime, upstream, seed=42)
    changed = build_trainer(runtime, upstream, seed=99)
    assert first.train_sampler.seed == 42
    assert list(first.train_sampler) == list(repeat.train_sampler)
    assert list(first.train_sampler) != list(changed.train_sampler)


def test_validation_loader_does_not_consume_global_torch_rng(runtime, upstream):
    trainer = build_trainer(runtime, upstream, val_rows=3)
    before = torch.get_rng_state().clone()
    list(trainer.val_dataloader)
    assert torch.equal(before, torch.get_rng_state())


def test_epoch_boundary_resume_does_not_restore_exhausted_iterator(
    runtime, upstream, monkeypatch
):
    first = build_trainer(runtime, upstream)
    iterator = iter(first.train_dataloader)
    for _ in range(first.steps_per_epoch):
        next(iterator)
    saved_loader = first.train_dataloader.state_dict()
    broken = build_trainer(runtime, upstream)
    broken.train_dataloader.load_state_dict(saved_loader)
    broken.train_sampler.set_epoch(1)
    assert len(list(broken.train_dataloader)) == 0  # The pinned upstream bug.

    resumed = build_trainer(runtime, upstream)
    loads = []
    inner = NS(
        resume_mode="resume_path",
        resume_from_path="global_step_3",
        engine=NS(load_checkpoint=lambda path: loads.append(path)),
        train_dataloader=resumed.train_dataloader,
    )
    monkeypatch.setattr(
        runtime, "validate_resume", lambda path, identity: (Path(path), 3)
    )
    handler = runtime.EpochCheckpointHandler(inner, {}, 3)
    resumed.resume_global_step = handler.load_checkpoint()
    resumed.fit()
    assert loads == ["global_step_3"]
    assert len(resumed.trained) == 12 and resumed.saved == [6]
    expected = build_trainer(runtime, upstream)
    expected.train_sampler.set_epoch(1)
    assert resumed.trained == list(expected.train_sampler)


def test_real_runtime_dependency_contract_when_requested():
    if os.getenv("RUN_REAL_SFT_CONTRACT") != "1":
        pytest.skip(
            "Local isolated source check; full dependency import is a separate CI job"
        )
    from tau2_agentic_rl.sft_contract import validate_runtime

    assert validate_runtime()["packages"]["transformers"] == "4.57.1"


@pytest.mark.skipif(
    os.getenv("RUN_REAL_SFT_CONTRACT") != "1",
    reason="Requires installed SFT veRL stack",
)
def test_real_hydra_accepts_launcher_overrides(monkeypatch, tmp_path):
    import sys

    from hydra import compose, initialize_config_module

    path = (
        Path(__file__).parents[2] / "training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py"
    )
    spec = importlib.util.spec_from_file_location("real_sft_launcher", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["sft", "--output-dir", str(tmp_path)])
    args = mod.parse_args()
    command = mod.build_training_command(args, tmp_path / "train.parquet", None)
    start = command.index(str(path.with_name("lora_sft_runtime.py"))) + 1
    with initialize_config_module(
        config_module="verl.trainer.config", version_base=None
    ):
        config = compose(config_name="sft_trainer_engine", overrides=command[start:])
    assert config.model.lora_rank == 64
    assert config.data.val_files is None
    assert config.trainer.test_freq == -1 and config.engine.seed == 42
    assert list(config.checkpoint.save_contents) == ["model", "optimizer", "extra"]


@pytest.mark.skipif(
    os.getenv("RUN_REAL_SFT_CONTRACT") != "1",
    reason="Requires installed SFT veRL stack",
)
def test_real_parquet_parent_dataset_and_qwen_masks(tmp_path):
    import json

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    path = (
        Path(__file__).parents[2] / "training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py"
    )
    spec = importlib.util.spec_from_file_location("real_sft_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = tmp_path / "input.jsonl"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a reservation",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "Follow policy."},
        {"role": "user", "content": "Look up ABC"},
        {
            "role": "assistant",
            "tool_calls": [{"name": "lookup", "arguments": {"id": "ABC"}}],
        },
        {"role": "tool", "content": "Found"},
        {"role": "tool", "content": "Confirmed"},
    ]
    answer = {"role": "assistant", "content": "Your reservation was found."}
    source.write_text(
        json.dumps(
            {
                "messages": messages,
                "answer": answer,
                "tools": tools,
                "metadata": {"source_dialog_id": "one", "turn_index": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train, _, _ = mod.prepare_dataset(source, tmp_path / "work", 0, 42, False)
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["QWEN_TOKENIZER_PATH"], local_files_only=True
    )
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "max_length": 16384,
            "pad_mode": "no_padding",
            "truncation": "error",
            "enable_thinking_default": False,
        }
    )
    dataset = mod.ARealLastAnswerSFTDataset(
        # Match Hydra's production path type; pinned veRL indexes path strings.
        parquet_files=str(train),
        tokenizer=tokenizer,
        config=config,
    )
    row = dataset[0]
    normalized = [mod.normalize_message(m) for m in messages]
    # Preparation deliberately stores tools_json with recursively sorted keys.
    # Qwen renders JSON in insertion order, so equal dictionaries can still
    # produce different token IDs. Assert semantic preservation independently,
    # then compare the exact canonical stream rather than weakening equality.
    assert json.loads(dataset.dataframe.iloc[0]["tools_json"]) == tools
    canonical_tools = json.loads(json.dumps(tools, sort_keys=True))
    prompt = tokenizer.apply_chat_template(
        normalized, tools=canonical_tools, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        normalized + [mod.normalize_message(answer)], tools=canonical_tools
    )
    assert row["input_ids"].tolist() == full
    assert not row["loss_mask"][: len(prompt)].any()
    assert row["loss_mask"][len(prompt) :].all()
    assert len(row["input_ids"]) <= 16384
    # A same-size manual edit to the prepared Parquet cannot bypass the cache.
    data = bytearray(train.read_bytes())
    data[-1] ^= 1
    train.write_bytes(data)
    with pytest.raises(RuntimeError, match="Parquet changed"):
        mod.prepare_dataset(source, tmp_path / "work", 0, 42, False)
