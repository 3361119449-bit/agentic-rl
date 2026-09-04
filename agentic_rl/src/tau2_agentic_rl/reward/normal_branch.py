"""Normal-resolution weighted reward branch."""

from __future__ import annotations

from tau2_agentic_rl.schemas import ComponentScore


def normalized_weighted_mean(
    components: dict[str, ComponentScore],
    weights: dict[str, float],
) -> float:
    """Omit inapplicable components and renormalize remaining weights."""
    active = {
        key: component for key, component in components.items() if component.applicable
    }
    if not active:
        return 0.0
    denominator = sum(weights[key] for key in active)
    if denominator <= 0:
        raise ValueError("active component weights must sum to a positive value")
    return sum(weights[key] * active[key].value for key in active) / denominator


def strict_success(components: dict[str, ComponentScore]) -> float:
    """Return one only if all applicable components equal one."""
    active = [
        component.value for component in components.values() if component.applicable
    ]
    return float(bool(active) and all(value == 1.0 for value in active))
