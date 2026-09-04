from pathlib import Path


def test_capped_trainer_keeps_the_8_by_3_by_8_contract() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "tau2_agentic_rl" / "verl_capped_trainer.py"
    ).read_text(encoding="utf-8")
    assert "conceptual_gen_batch_size = 8" in source
    assert "max_num_gen_batches = 3" in source
    assert "rollout_group_size = 8" in source
    assert "skipped_optimizer_steps" in source
