import ast
from pathlib import Path
from types import SimpleNamespace


def test_trainer_starts_with_32_slots_and_requests_four_groups() -> None:
    path = (
        Path(__file__).parents[1] / "src" / "tau2_agentic_rl" / "verl_capped_trainer.py"
    )
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    original = next(node for node in parsed.body
                    if isinstance(node, ast.ClassDef) and node.name == "CappedPPOTrainerSync")
    wrapper = ast.parse("class UnderTest:\n    pass\n")
    wrapper.body[0].body = [node for node in original.body if isinstance(node, ast.Assign)
                           or isinstance(node, ast.FunctionDef) and node.name == "step"]
    ast.fix_missing_locations(wrapper)
    scope = {"KVBatchMeta": object}
    exec(compile(wrapper, str(path), "exec"), scope)
    trainer = scope["UnderTest"]()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(critic_warmup=0))
    trainer.parameter_sync_step = 1
    events = []
    trainer.replay_buffer = SimpleNamespace(begin_attempt=lambda: events.append("begin"))
    trainer._add_prompts_to_generate = lambda count: events.append(("generate", count * trainer.rollout_group_size))
    trainer._step_once = lambda metrics, timing, **kw: events.append(("sample", kw["sample_batch_size"]))
    trainer.step({}, {})
    assert events == ["begin", ("generate", 32), ("sample", 4)]


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
    isolates = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_isolates_failed_groups")
    wrapper.body[0].body = [method, isolates]
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
