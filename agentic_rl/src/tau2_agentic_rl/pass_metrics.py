"""Standard-library-only contracts shared by the two Tau2 pass^k reporters."""

import math
import random

# data/tau2/domains/airline/split_tasks.json at this exact upstream revision.
TAU2_COMMIT = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
TAU2_OFFICIAL_MAX_STEPS = 200
TAU2_OFFICIAL_MAX_ERRORS = 10
TAU2_OFFICIAL_SEED = 300
TAU2_OFFICIAL_AGENT_TEMPERATURE = 0.0
TAU2_OFFICIAL_USER_TEMPERATURE = 0.0
OFFICIAL_EVALUATION_PROTOCOL_SAMPLES = {
    "pass1": 1,
    "pass1_pass4": 4,
}


def official_trial_seeds(seed: int, num_trials: int) -> list[int]:
    """Match pinned Tau2 batch.py: one deterministic seed per trial."""
    if type(seed) is not int or type(num_trials) is not int or num_trials < 1:
        raise ValueError("seed must be an integer and num_trials must be positive")
    generator = random.Random(seed)
    return [generator.randint(0, 1_000_000) for _ in range(num_trials)]


OFFICIAL_TEST_IDS = frozenset(
    {
        "2",
        "6",
        "8",
        "13",
        "16",
        "18",
        "19",
        "22",
        "24",
        "25",
        "26",
        "29",
        "30",
        "31",
        "32",
        "35",
        "37",
        "44",
        "45",
        "48",
    }
)


def validate_official_test_ids(task_ids, tau2_commit):
    """A self-consistent manifest hash is not proof of the official task split."""
    if tau2_commit != TAU2_COMMIT:
        raise ValueError("official test requires the pinned Tau2 commit")
    if (
        not isinstance(task_ids, (list, tuple))
        or any(type(task) is not str for task in task_ids)
        or len(task_ids) != len(OFFICIAL_TEST_IDS)
        or set(task_ids) != OFFICIAL_TEST_IDS
    ):
        raise ValueError("tasks must be exactly the 20 official test IDs")


def validate_unit_score(value, *, name="official reward"):
    """Reject corrupt JSON scores, including bool (a subclass of int)."""
    if (
        type(value) not in (int, float)
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite non-boolean number in [0, 1]")
    return value


def official_success(reward):
    """Use pinned Tau2's inclusive interval, including its float boundaries."""
    reward = validate_unit_score(reward)
    return (1 - 1e-6) <= reward <= (1 + 1e-6)
