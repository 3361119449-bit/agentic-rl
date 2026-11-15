from tau2_agentic_rl.budget import ContextBudget


def test_budget_is_checked_before_generation() -> None:
    budget = ContextBudget(max_context_tokens=100)
    assert budget.decide(60).max_new_tokens == 40
    assert budget.decide(99).max_new_tokens == 1
    decision = budget.decide(100)
    assert decision.can_generate is False
    assert decision.max_new_tokens == 0
