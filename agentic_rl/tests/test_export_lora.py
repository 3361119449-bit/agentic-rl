from pathlib import Path

from scripts.export_verl_lora import build_command


def test_export_uses_pinned_verl_model_merger_interface() -> None:
    command = build_command(Path("C:/global_step_10"), Path("C:/export"))
    assert command[1:5] == ["-m", "verl.model_merger", "merge", "--backend"]
    assert "--local_dir" in command
    assert "--target_dir" in command
