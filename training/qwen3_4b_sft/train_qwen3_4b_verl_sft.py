#!/usr/bin/env python3
"""Prepare airline prefix/answer SFT data and launch Qwen3-4B SFT with verl.

The source JSONL stores the prompt/history in ``messages`` and the supervised
next assistant turn in ``answer``. This module appends ``answer`` during
preparation and provides a verl dataset class whose loss mask covers only that
last assistant turn.

Recommended runtime: verl v0.7.1 and transformers >= 4.51.0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
except ModuleNotFoundError as exc:
    if exc.name != "verl":
        raise
    # Keep --help and source-only checks usable before verl is installed.
    MultiTurnSFTDataset = object  # type: ignore[assignment,misc]


SCRIPT_VERSION = 4
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DATA_RELATIVE_PATH = Path(
    "datasets/tau2_airline_sft_strict_cleaned/data/"
    "airline_sft_no_thinking_under_16k_strict_leakage_cleaned.jsonl"
)
FORBIDDEN_KEYS = {"thinking", "reasoning", "reasoning_content"}
THINK_TAG_RE = re.compile(r"</?think(?:\s[^>]*)?>", re.IGNORECASE)


class ARealLastAnswerSFTDataset(MultiTurnSFTDataset):
    """Tokenize the full conversation while supervising only the final answer.

    verl's stock ``MultiTurnSFTDataset`` supervises every assistant message and
    tokenizes each turn separately. AReaL rows are prefix/next-answer pairs,
    and Qwen's template groups consecutive tool responses. Whole-conversation
    tokenization preserves the exact Qwen template and avoids training all
    historical assistant turns again.
    """

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional
        from verl.utils.py_functional import convert_nested_value_to_list_recursive

        row = self.dataframe.iloc[item].to_dict()
        messages = convert_nested_value_to_list_recursive(row[self.messages_key])
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Row {item}: expected at least two messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Row {item}: final message must be the appended answer")

        tools = None
        tools_json = row.get("tools_json")
        if isinstance(tools_json, str) and tools_json.strip() not in {"", "null"}:
            tools = json.loads(tools_json)
        elif self.tools is not None:
            tools = convert_nested_value_to_list_recursive(self.tools[item])

        template_kwargs = dict(self.apply_chat_template_kwargs)
        enable_thinking = (
            self.enable_thinking[item]
            if self.enable_thinking is not None
            else self.enable_thinking_default
        )
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(enable_thinking)

        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        full = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )

        prompt_ids = prompt["input_ids"][0]
        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        prompt_length = prompt_ids.shape[0]
        if prompt_length >= input_ids.shape[0]:
            raise ValueError(f"Row {item}: final answer produced no target tokens")
        if not torch.equal(input_ids[:prompt_length], prompt_ids):
            raise ValueError(
                f"Row {item}: Qwen chat-template prompt is not a prefix of the full sample"
            )

        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[prompt_length:] = 1
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"Row {item}: {sequence_length=} exceeds {self.max_length=}"
                )
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                position_ids = position_ids[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
            else:
                raise ValueError(f"Unknown truncation method: {self.truncation}")

        if not torch.any(loss_mask):
            raise ValueError(f"Row {item}: truncation removed the complete answer")

        if self.pad_mode == "right":
            padding = self.max_length - input_ids.shape[0]
            if padding > 0:
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = functional.pad(input_ids, (0, padding), value=pad_id)
                attention_mask = functional.pad(attention_mask, (0, padding), value=0)
                loss_mask = functional.pad(loss_mask, (0, padding), value=0)
                position_ids = functional.pad(position_ids, (0, padding), value=0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }

        if self.pad_mode == "no_padding":
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        raise ValueError(f"Unknown pad mode: {self.pad_mode}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(model: str, revision: str, max_length: int) -> dict[str, Any]:
    """Fingerprint tokenizer/template inputs that affect final SFT tokenization."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=False
    )
    chat_template = str(tokenizer.chat_template or "")
    return {
        "model": model,
        "requested_revision": revision,
        "resolved_revision": tokenizer.init_kwargs.get("_commit_hash", revision),
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "transformers_version": importlib.metadata.version("transformers"),
        "max_length": max_length,
        "truncation": "error",
        "loss_scope": "last_assistant_answer_only",
        "thinking_reasoning_rejected": True,
        "normalization_version": SCRIPT_VERSION,
    }


def resolve_model_snapshot(model: str, revision: str, *, metadata_only: bool = False) -> str:
    """Resolve both training weights and tokenizer to a single immutable snapshot."""
    local = Path(model)
    if local.is_dir():
        return str(local.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model,
        revision=revision,
        allow_patterns=(
            ["*.json", "*.jinja", "*.txt", "*.model"] if metadata_only else None
        ),
    )


def prepared_cache_matches(
    manifest: dict[str, Any],
    *,
    source: Path,
    source_sha256: str,
    val_ratio: float,
    seed: int,
    preprocessing_identity: dict[str, Any],
    val_path: Path | None,
) -> bool:
    """Require content and tokenizer identity, not only source file size."""
    return bool(
        manifest.get("script_version") == SCRIPT_VERSION
        and manifest.get("input") == str(source)
        and manifest.get("input_size") == source.stat().st_size
        and manifest.get("input_sha256") == source_sha256
        and manifest.get("val_ratio") == val_ratio
        and manifest.get("seed") == seed
        and manifest.get("preprocessing_identity") == preprocessing_identity
        and (val_path is None or val_path.exists())
    )


def reject_thinking(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden field {key!r} at {path}")
            reject_thinking(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_thinking(child, f"{path}[{index}]")
    elif isinstance(value, str) and THINK_TAG_RE.search(value):
        raise ValueError(f"Forbidden <think> tag at {path}")


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, str]] | None:
    if not tool_calls:
        return None
    normalized: list[dict[str, str]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("tool_calls entries must be JSON objects")
        function = (
            call.get("function") if isinstance(call.get("function"), dict) else call
        )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every tool call must have a non-empty name")
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("Each message must be a JSON object")
    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"Unsupported message role: {role!r}")
    content = message.get("content", "")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    name = message.get("name")
    if name is not None and not isinstance(name, str):
        name = str(name)
    return {
        "role": role,
        "content": content,
        "name": name,
        "tool_calls": normalize_tool_calls(message.get("tool_calls")),
    }


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], bytes]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number}: top-level value must be an object"
                )
            yield line_number, record, raw_line


def source_dialog_id(record: dict[str, Any], line_number: int) -> str:
    metadata = record.get("metadata")
    value = metadata.get("source_dialog_id") if isinstance(metadata, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Line {line_number}: missing metadata.source_dialog_id")
    return value


def choose_validation_dialogs(dialogs: set[str], ratio: float, seed: int) -> set[str]:
    if ratio <= 0:
        return set()
    count = max(1, round(len(dialogs) * ratio))
    count = min(count, max(0, len(dialogs) - 1))
    ranked = sorted(
        dialogs,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    return set(ranked[:count])


def scan_source(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    dialogs: set[str] = set()
    rows = 0
    targets = {"text": 0, "tool_call": 0}
    rows_with_tools = 0
    for line_number, record, raw_line in read_jsonl(path):
        digest.update(raw_line)
        reject_thinking(record, f"line[{line_number}]")
        messages = record.get("messages")
        answer = record.get("answer")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Line {line_number}: messages must be a non-empty list")
        if not isinstance(answer, dict) or answer.get("role") != "assistant":
            raise ValueError(f"Line {line_number}: answer must be an assistant message")
        if messages[-1].get("role") not in {"user", "tool"}:
            raise ValueError(f"Line {line_number}: history must end in user/tool")
        normalized_answer = normalize_message(answer)
        if (
            not normalized_answer["content"].strip()
            and not normalized_answer["tool_calls"]
        ):
            raise ValueError(f"Line {line_number}: empty supervised answer")
        target_kind = "tool_call" if normalized_answer["tool_calls"] else "text"
        targets[target_kind] += 1
        tools = record.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not tools:
                raise ValueError(f"Line {line_number}: tools must be a non-empty list")
            rows_with_tools += 1
        dialogs.add(source_dialog_id(record, line_number))
        rows += 1
    if rows == 0:
        raise ValueError("The input JSONL is empty")
    return {
        "rows": rows,
        "dialogs": dialogs,
        "sha256": digest.hexdigest(),
        "targets": targets,
        "rows_with_tools": rows_with_tools,
    }


def prepare_dataset(
    source: Path,
    work_dir: Path,
    val_ratio: float,
    seed: int,
    force: bool,
    preprocessing_identity: dict[str, Any] | None = None,
) -> tuple[Path, Path | None, dict[str, Any]]:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from exc

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SFT JSONL not found: {source}")
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")

    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = work_dir / "train.parquet"
    val_path = work_dir / "validation.parquet" if val_ratio > 0 else None
    manifest_path = work_dir / "prepare_manifest.json"
    source_sha256 = sha256_file(source)
    expected_identity = dict(preprocessing_identity or {})

    if not force and train_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reusable = prepared_cache_matches(
            manifest,
            source=source,
            source_sha256=source_sha256,
            val_ratio=val_ratio,
            seed=seed,
            preprocessing_identity=expected_identity,
            val_path=val_path,
        )
        if reusable:
            print(f"Reusing prepared dataset in {work_dir}")
            return train_path, val_path, manifest
        raise RuntimeError("Prepared files are stale; rerun with --force-prepare")

    scan = scan_source(source)
    validation_dialogs = choose_validation_dialogs(scan["dialogs"], val_ratio, seed)

    tool_call_type = arrow.struct(
        [arrow.field("name", arrow.string()), arrow.field("arguments", arrow.string())]
    )
    message_type = arrow.struct(
        [
            arrow.field("role", arrow.string(), nullable=False),
            arrow.field("content", arrow.string(), nullable=False),
            arrow.field("name", arrow.string()),
            arrow.field("tool_calls", arrow.list_(tool_call_type)),
        ]
    )
    schema = arrow.schema(
        [
            arrow.field("messages", arrow.list_(message_type), nullable=False),
            arrow.field("tools_json", arrow.string(), nullable=False),
            arrow.field("sample_id", arrow.string(), nullable=False),
            arrow.field("source_dialog_id", arrow.string(), nullable=False),
            arrow.field("turn_index", arrow.int32(), nullable=False),
        ]
    )

    temp_train = work_dir / "train.parquet.tmp"
    temp_val = work_dir / "validation.parquet.tmp"
    for temporary in (temp_train, temp_val):
        if temporary.exists():
            temporary.unlink()

    writers: dict[str, Any] = {
        "train": parquet.ParquetWriter(temp_train, schema, compression="zstd")
    }
    if val_path is not None:
        writers["validation"] = parquet.ParquetWriter(
            temp_val, schema, compression="zstd"
        )
    buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in writers}
    counts = {name: 0 for name in writers}

    def flush(split: str) -> None:
        if buffers[split]:
            writers[split].write_table(
                arrow.Table.from_pylist(buffers[split], schema=schema)
            )
            buffers[split].clear()

    try:
        for line_number, record, _ in read_jsonl(source):
            dialog_id = source_dialog_id(record, line_number)
            split = "validation" if dialog_id in validation_dialogs else "train"
            metadata = record.get("metadata") or {}
            turn_index = int(metadata.get("turn_index", -1))
            messages = [normalize_message(message) for message in record["messages"]]
            messages.append(normalize_message(record["answer"]))
            buffers[split].append(
                {
                    "messages": messages,
                    "tools_json": json.dumps(
                        record.get("tools"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "sample_id": f"{dialog_id}:{turn_index}:{line_number}",
                    "source_dialog_id": dialog_id,
                    "turn_index": turn_index,
                }
            )
            counts[split] += 1
            if len(buffers[split]) >= 256:
                flush(split)
        for split in writers:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()

    os.replace(temp_train, train_path)
    if val_path is not None:
        os.replace(temp_val, val_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "input": str(source),
        "input_size": source.stat().st_size,
        "input_sha256": scan["sha256"],
        "preprocessing_identity": expected_identity,
        "source_rows": scan["rows"],
        "source_dialogs": len(scan["dialogs"]),
        "target_types": scan["targets"],
        "rows_with_tool_schemas": scan["rows_with_tools"],
        "val_ratio": val_ratio,
        "seed": seed,
        "train_rows": counts["train"],
        "validation_rows": counts.get("validation", 0),
        "train_dialogs": len(scan["dialogs"] - validation_dialogs),
        "validation_dialogs": len(validation_dialogs),
        "loss_scope": "last_assistant_answer_only",
        "thinking_reasoning_rejected": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return train_path, val_path, manifest


def numeric_version(package: str) -> tuple[int, int, int]:
    raw = importlib.metadata.version(package)
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
    if not match:
        raise RuntimeError(f"Cannot parse {package} version: {raw}")
    return tuple(int(value or 0) for value in match.groups())  # type: ignore[return-value]


def validate_training_runtime() -> None:
    for package, minimum in (("verl", (0, 7, 1)), ("transformers", (4, 51, 0))):
        try:
            installed = numeric_version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing package {package}; see the training README"
            ) from exc
        if installed < minimum:
            raise RuntimeError(
                f"{package}>={'.'.join(map(str, minimum))} is required; found "
                f"{importlib.metadata.version(package)}"
            )


def build_training_command(
    args: argparse.Namespace, train_path: Path, val_path: Path | None
) -> list[str]:
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    if args.global_batch_size < 1 or args.micro_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_length < 1 or args.max_token_len_per_gpu < 1:
        raise ValueError("Token limits must be positive")
    if args.lora_rank < 0:
        raise ValueError("--lora-rank cannot be negative")
    if args.ulysses_size < 1 or args.num_gpus % args.ulysses_size:
        raise ValueError("--ulysses-size must divide --num-gpus")
    data_parallel_size = args.num_gpus // args.ulysses_size
    if args.global_batch_size % data_parallel_size:
        raise ValueError(
            "--global-batch-size must divide evenly across data-parallel ranks"
        )
    if args.max_token_len_per_gpu < args.max_length and args.ulysses_size == 1:
        raise ValueError(
            "--max-token-len-per-gpu must be >= --max-length without sequence parallelism"
        )

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if args.lora_rank > 0 else 2e-5

    custom_module = Path(__file__).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    val_override = str(val_path.resolve()) if val_path is not None else "null"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.num_gpus}",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={train_path.resolve()}",
        f"data.val_files={val_override}",
        f"data.train_batch_size={args.global_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size}",
        "data.messages_key=messages",
        "data.pad_mode=no_padding",
        f"data.max_length={args.max_length}",
        "data.truncation=error",
        "data.use_dynamic_bsz=True",
        f"data.max_token_len_per_gpu={args.max_token_len_per_gpu}",
        f"data.num_workers={args.num_workers}",
        "data.ignore_input_ids_mismatch=False",
        f"data.custom_cls.path={custom_module}",
        "data.custom_cls.name=ARealLastAnswerSFTDataset",
        f"model.path={args.model}",
        "model.trust_remote_code=False",
        "model.enable_gradient_checkpointing=True",
        "model.use_remove_padding=True",
        f"model.lora_rank={args.lora_rank}",
        f"model.lora_alpha={args.lora_alpha}",
        "model.target_modules=all-linear",
        "engine=fsdp",
        f"engine.strategy={args.fsdp_strategy}",
        f"engine.ulysses_sequence_parallel_size={args.ulysses_size}",
        f"engine.param_offload={str(args.param_offload).lower()}",
        f"engine.optimizer_offload={str(args.optimizer_offload).lower()}",
        f"optim.lr={learning_rate}",
        f"optim.lr_warmup_steps_ratio={args.warmup_ratio}",
        f"optim.weight_decay={args.weight_decay}",
        "optim.lr_scheduler_type=cosine",
        f"trainer.default_local_dir={output_dir}",
        "trainer.project_name=tau2-airline-sft",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.total_epochs={args.epochs}",
        "trainer.save_freq=after_each_epoch",
        "trainer.test_freq=after_each_epoch",
        "trainer.logger=[console]",
        f"trainer.seed={args.seed}",
        f"trainer.n_gpus_per_node={args.num_gpus}",
        f"trainer.resume_mode={args.resume_mode}",
        "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
    ]
    if args.resume_mode == "resume_path":
        if args.resume_from_path is None:
            raise ValueError(
                "--resume-from-path is required with --resume-mode resume_path"
            )
        command.append(f"trainer.resume_from_path={args.resume_from_path.resolve()}")
    command.extend(args.extra_config)
    return command


def _safe_run_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not normalized:
        raise ValueError("run name is empty after normalization")
    return normalized


def resolve_sft_run(args: argparse.Namespace) -> tuple[str, Path]:
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if args.lora_rank > 0 else 2e-5
    mode = f"lora-r{args.lora_rank}" if args.lora_rank else "full"
    generated = (
        f"{args.experiment_name}-{mode}-lr{learning_rate:g}-"
        f"ep{args.epochs}-seed{args.seed}"
    )
    run_name = _safe_run_name(args.run_name or generated)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            repository_root() / "training/qwen3_4b_sft/runs" / run_name / "checkpoints"
        )
    return run_name, output_dir


def parse_args() -> argparse.Namespace:
    root = repository_root()
    default_work = root / "training/qwen3_4b_sft/work"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-jsonl", type=Path, default=root / DATA_RELATIVE_PATH)
    parser.add_argument("--work-dir", type=Path, default=default_work)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--num-gpus", type=int, default=int(os.getenv("NUM_GPUS", "1")))
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16_384)
    parser.add_argument("--max-token-len-per-gpu", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--ulysses-size", type=int, default=1)
    parser.add_argument("--fsdp-strategy", choices=("fsdp", "fsdp2"), default="fsdp")
    parser.add_argument("--param-offload", action="store_true")
    parser.add_argument("--optimizer-offload", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="qwen3-4b-instruct-2507-full-sft")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume-mode", choices=("disable", "resume_path"), default="disable"
    )
    parser.add_argument("--resume-from-path", type=Path)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        metavar="HYDRA_OVERRIDE",
        help="Append a raw verl/Hydra override; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name, output_dir = resolve_sft_run(args)
    args.experiment_name = run_name
    args.output_dir = output_dir
    if (
        not args.prepare_only
        and not args.dry_run
        and args.resume_mode == "disable"
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose --run-name or "
            "use --resume-mode resume_path --resume-from-path ..."
        )
    requested_model = args.model
    args.model = resolve_model_snapshot(
        args.model, args.model_revision, metadata_only=args.prepare_only or args.dry_run
    )
    identity = tokenizer_identity(args.model, args.model_revision, args.max_length)
    identity["requested_model"] = requested_model
    if Path(args.model).parent.name == "snapshots":
        identity["resolved_revision"] = Path(args.model).name
    train_path, val_path, _ = prepare_dataset(
        source=args.data_jsonl,
        work_dir=args.work_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force_prepare,
        preprocessing_identity=identity,
    )
    if args.prepare_only:
        return
    validate_training_runtime()
    command = build_training_command(args, train_path, val_path)
    print("Launching:\n" + shlex.join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
