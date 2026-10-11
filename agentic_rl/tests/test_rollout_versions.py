"""Weight versions must survive the production multi-turn loop and TQ contract."""

import pytest
from test_procredit_rollout import rollout_fixture
from test_rollout_integration import production_loop_class

from tau2_agentic_rl.slot_recovery import RolloutInfrastructureError


@pytest.mark.parametrize("version", [0, 1, 6])
@pytest.mark.parametrize("server_bounds", [True, False])
def test_generated_weight_version_is_exported(scratch_dir, version, server_bounds):
    fields = ({"min_global_steps": version, "max_global_steps": version}
              if server_bounds else {"global_steps": version})
    _, output, record = rollout_fixture(scratch_dir, version_fields=[fields] * 3)
    assert output.extra_fields["min_global_steps"] == version
    assert output.extra_fields["max_global_steps"] == version
    assert type(output.extra_fields["min_global_steps"]) is int
    assert type(output.extra_fields["max_global_steps"]) is int
    assert output.extra_fields["policy_version"] == record.policy_version == version


def test_all_generation_rounds_contribute_bounds(scratch_dir):
    _, output, _ = rollout_fixture(scratch_dir, version_fields=[
        {"min_global_steps": 1, "max_global_steps": 2},
        {"min_global_steps": 0, "max_global_steps": 2},
        {"min_global_steps": 2, "max_global_steps": 2},
    ])
    assert output.extra_fields["min_global_steps"] == 0
    assert output.extra_fields["max_global_steps"] == 2


def test_unknown_version_fails_before_environment_delivery_or_scoring(scratch_dir):
    with pytest.raises(RolloutInfrastructureError, match="policy_version_alignment"):
        rollout_fixture(scratch_dir, version_fields=[{}] * 3)


@pytest.mark.parametrize("fields, expected", [
    ({"min_global_steps": 3, "max_global_steps": 5, "global_steps": 5}, (3, 5)),
    ({"min_global_steps": None, "max_global_steps": None, "global_steps": 6}, (6, 6)),
    ({"min_global_steps": 0, "max_global_steps": 0}, (0, 0)),
])
def test_version_bounds_use_server_metadata(fields, expected):
    _, scope = production_loop_class()
    assert scope["_rollout_version_bounds"](fields) == expected


@pytest.mark.parametrize("fields", [
    {}, {"min_global_steps": 1}, {"max_global_steps": 1},
    {"global_steps": None}, {"global_steps": -1}, {"global_steps": True},
    {"global_steps": 1.5}, {"global_steps": "1"},
    {"min_global_steps": 2, "max_global_steps": 1},
])
def test_unverified_or_invalid_versions_are_not_coerced_to_zero(fields):
    _, scope = production_loop_class()
    with pytest.raises(ValueError, match="weight version"):
        scope["_rollout_version_bounds"](fields)
