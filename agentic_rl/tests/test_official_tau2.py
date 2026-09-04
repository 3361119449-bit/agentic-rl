from tau2_agentic_rl.reward.official_tau2 import parse_official_reward_info


def test_parses_actual_tau2_reward_info_field_names() -> None:
    result = parse_official_reward_info(
        0.0,
        {
            "reward_basis": ["DB", "COMMUNICATE"],
            "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 0.0},
            "db_check": {"db_match": True, "db_reward": 1.0},
            "communicate_checks": [
                {"info": "10", "met": True},
                {"info": "20", "met": False},
            ],
        },
    )
    assert result.db_applicable is True
    assert result.db_score == 1.0
    assert result.communicate_partial == 0.5
    assert result.communicate_all is False
