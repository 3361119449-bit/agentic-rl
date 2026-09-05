import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_sft_module():
    path = (
        Path(__file__).parents[2] / "training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py"
    )
    spec = importlib.util.spec_from_file_location("airline_sft_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_size_changed_source_does_not_reuse_cache() -> None:
    module = _load_sft_module()
    source = Path(__file__)
    identity = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "resolved_revision": "abc",
        "chat_template_sha256": "template",
    }
    manifest = {
        "script_version": module.SCRIPT_VERSION,
        "input": str(source.resolve()),
        "input_size": source.stat().st_size,
        "input_sha256": "old-content",
        "val_ratio": 0.02,
        "seed": 42,
        "preprocessing_identity": identity,
    }
    assert not module.prepared_cache_matches(
        manifest,
        source=source.resolve(),
        source_sha256="new-content",
        val_ratio=0.02,
        seed=42,
        preprocessing_identity=identity,
        val_path=None,
    )


def test_sft_run_directory_encodes_hyperparameters() -> None:
    module = _load_sft_module()
    run_name, output = module.resolve_sft_run(
        Namespace(
            learning_rate=1e-4,
            lora_rank=64,
            experiment_name="airline",
            epochs=2,
            seed=42,
            run_name=None,
            output_dir=None,
        )
    )
    assert run_name == "airline-lora-r64-lr0.0001-ep2-seed42"
    assert output.parts[-2:] == (run_name, "checkpoints")


def test_sft_revision_resolves_full_training_snapshot(monkeypatch, scratch_dir):
    import sys
    from types import SimpleNamespace

    module = _load_sft_module()
    requests = []

    def snapshot_download(**kwargs):
        requests.append(kwargs)
        return str(scratch_dir / "snapshots" / "pinned-sha")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    path = module.resolve_model_snapshot("Qwen/Qwen3-4B-Instruct-2507", "pinned-sha")
    assert Path(path).name == "pinned-sha"
    assert requests[0]["revision"] == "pinned-sha"
    assert requests[0].get("allow_patterns") is None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'f"model.path={args.model}"' in source
