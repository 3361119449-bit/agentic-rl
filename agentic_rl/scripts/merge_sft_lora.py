"""Merge an Airline SFT LoRA into Qwen3 before creating a fresh RL LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
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
    for name in ("config.json", "tokenizer_config.json"):
        if not (args.output / name).is_file():
            raise RuntimeError(f"merged model output is incomplete: {name}")
    print(args.output)


if __name__ == "__main__":
    main()
