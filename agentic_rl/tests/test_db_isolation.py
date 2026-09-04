from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter


def test_database_hash_snapshot_is_instance_local() -> None:
    left = Tau2GymAdapter(task_id="0", user_model="mock", user_cache_dir="cache-a")
    right = Tau2GymAdapter(task_id="0", user_model="mock", user_cache_dir="cache-b")
    left._initial_db_hash = "left"
    right._initial_db_hash = "right"
    assert left.initial_db_hash() == "left"
    assert right.initial_db_hash() == "right"
