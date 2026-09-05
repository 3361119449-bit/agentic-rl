"""Compare an SFT PEFT adapter with its merged model on a fixed input.

Run this on the GPU host after ``export_verl_lora.py`` and
``merge_sft_lora.py``.  The models are loaded sequentially so the check does
not require enough memory for two copies of Qwen3 at once.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from tau2_agentic_rl.base_identity import model_files, validate_adapter_base
from tau2_agentic_rl.versions import sha256_file, sha256_json

DEFAULT_PROMPT = (
    "A customer asks whether a cancelled flight can be refunded. "
    "Reply briefly and do not invent reservation details."
)


def _require_adapter(path: Path) -> None:
    required = (
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError("incomplete PEFT adapter: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()
    _require_adapter(args.adapter)
    args.base_model, identity = validate_adapter_base(args.adapter, args.base_model)
    provenance = json.loads(
        (args.merged_model / "merge_provenance.json").read_text(encoding="utf-8")
    )
    if provenance != {
        "base_identity_sha256": sha256_json(identity),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "merged_files": model_files(args.merged_model),
    }:
        raise ValueError("merged model provenance mismatch")
    print("training base and merge provenance identity verified")
    if not (args.merged_model / "config.json").is_file():
        raise FileNotFoundError(f"incomplete merged model: {args.merged_model}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("this equivalence check requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(
        args.merged_model, trust_remote_code=False
    )
    messages = [{"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")

    def last_logits(model) -> torch.Tensor:
        model.eval()
        with torch.inference_mode():
            return model(**inputs).logits[0, -1].float().cpu()

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
    )
    adapter_model = PeftModel.from_pretrained(base, args.adapter)
    adapter_logits = last_logits(adapter_model)
    del adapter_model, base
    gc.collect()
    torch.cuda.empty_cache()

    merged = AutoModelForCausalLM.from_pretrained(
        args.merged_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
    )
    merged_logits = last_logits(merged)
    max_abs_diff = (adapter_logits - merged_logits).abs().max().item()
    torch.testing.assert_close(
        adapter_logits,
        merged_logits,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(
        "adapter/merged logits match: "
        f"max_abs_diff={max_abs_diff:.6g}, atol={args.atol}, rtol={args.rtol}"
    )


if __name__ == "__main__":
    main()
