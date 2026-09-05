import ast
from pathlib import Path


def test_capped_trainer_keeps_the_8_by_3_by_8_contract() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "tau2_agentic_rl" / "verl_capped_trainer.py"
    ).read_text(encoding="utf-8")
    assert "conceptual_gen_batch_size = 8" in source
    assert "max_num_gen_batches = 3" in source
    assert "rollout_group_size = 8" in source
    assert "skipped_optimizer_steps" in source


def test_live_eviction_override_excludes_partially_failed_training_groups():
    path = Path(__file__).parents[1] / "src/tau2_agentic_rl/verl_capped_trainer.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "CappedDynamicReplayBuffer"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_terminal_eviction_reasons"
    )
    # Execute the actual production method with only its veRL parent mocked.
    # No duplicate implementation, Ray, torch, or GPU needed in this unit test.
    wrapper = ast.parse("class UnderTest(FakeParent):\n    pass\n")
    wrapper.body[0].body = [method]
    ast.fix_missing_locations(wrapper)

    class FakeParent:
        failure_keys = {
            "train": {"empty_failure", "seven_success_one_failure"},
            "val": {"validation_failure"},
        }

        def _terminal_eviction_reasons(self, step, partition):
            return (
                set(),
                set(),
                {"empty_failure"} if partition == "train" else set(),
                {},
            )

    scope = {"FakeParent": FakeParent}
    exec(compile(wrapper, str(path), "exec"), scope)
    buffer = scope["UnderTest"]()
    assert buffer._terminal_eviction_reasons(1, "train")[2] == {
        "empty_failure",
        "seven_success_one_failure",
    }
    assert buffer._terminal_eviction_reasons(1, "val")[2] == set()
