"""Merge an Airline SFT LoRA into Qwen3 before creating a fresh RL LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2_agentic_rl.base_identity import model_files, validate_adapter_base
from tau2_agentic_rl.versions import sha256_file, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Relocated identical local snapshot; default: training snapshot",
    )
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.sft_adapter / "adapter_config.json",
        args.sft_adapter / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "--sft-adapter must be a standard PEFT directory exported by "
            "verl.model_merger; missing: " + ", ".join(missing)
        )

    args.base_model, identity = validate_adapter_base(args.sft_adapter, args.base_model)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    merged = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    tokenizer.save_pretrained(args.output)
    (args.output / "merge_provenance.json").write_text(
        json.dumps(
            {
                "base_identity_sha256": sha256_json(identity),
                "adapter_sha256": sha256_file(
                    args.sft_adapter / "adapter_model.safetensors"
                ),
                "adapter_config_sha256": sha256_file(
                    args.sft_adapter / "adapter_config.json"
                ),
                "merged_files": model_files(args.output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in ("config.json", "tokenizer_config.json"):
        if not (args.output / name).is_file():
            raise RuntimeError(f"merged model output is incomplete: {name}")
    print(args.output)


if __name__ == "__main__":
    main()
