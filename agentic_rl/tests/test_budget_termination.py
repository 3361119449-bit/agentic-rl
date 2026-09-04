from tau2_agentic_rl.budget import ContextBudget


def test_budget_is_checked_before_generation() -> None:
    budget = ContextBudget(
        max_context_tokens=100,
        reserved_observation_tokens=10,
        reserved_template_tokens=5,
        min_final_response_tokens=8,
        per_turn_max_new_tokens=20,
    )
    assert budget.decide(60).max_new_tokens == 20
    decision = budget.decide(80)
    assert decision.can_generate is False
    assert decision.max_new_tokens == 0
