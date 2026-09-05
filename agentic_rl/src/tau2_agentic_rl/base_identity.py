"""Content-bound base identity shared by training, export, merge and verification."""

import json
from pathlib import Path

from tau2_agentic_rl.versions import sha256_file, sha256_json

MANIFEST = "base_model_identity.json"


def model_files(path):
    path = Path(path)
    names = {
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    }
    return {
        p.relative_to(path).as_posix(): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and (
            p.name in names
            or p.name.startswith("tokenizer")
            or p.name.startswith("model")
            and p.suffix in {".json", ".safetensors", ".bin"}
            or p.suffix == ".jinja"
        )
    }


def capture_base_identity(path):
    path = Path(path).resolve()
    files = model_files(path)
    if (
        "config.json" not in files
        or "tokenizer_config.json" not in files
        or not any(key.endswith((".safetensors", ".bin")) for key in files)
    ):
        raise ValueError("base identity requires a complete local model and tokenizer")
    config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = {
        "inline": config.get("chat_template"),
        "files": {k: v for k, v in files.items() if k.endswith(".jinja")},
    }
    return {
        "schema_version": 1,
        "snapshot_path": str(path),
        "revision": path.name if path.parent.name == "snapshots" else None,
        "files": files,
        "chat_template_sha256": sha256_json(template),
    }


def save_base_identity(directory, identity):
    path = Path(directory) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("saved base identity differs; refusing to overwrite")
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2)
    return path


def find_base_identity(checkpoint):
    candidate = Path(checkpoint).resolve()
    for _ in range(5):
        if (candidate / MANIFEST).is_file():
            return candidate / MANIFEST
        candidate = candidate.parent
    raise FileNotFoundError(
        "training base identity missing; supply the original --base-identity, never infer from today's Hub main"
    )


def validate_adapter_base(adapter, base_model=None):
    identity = json.loads((Path(adapter) / MANIFEST).read_text(encoding="utf-8"))
    path = Path(base_model or identity["snapshot_path"]).resolve()
    actual = capture_base_identity(path)
    for key in ("schema_version", "files", "chat_template_sha256"):
        if identity.get(key) != actual[key]:
            raise ValueError(f"adapter training base identity mismatch: {key}")
    return path, identity
