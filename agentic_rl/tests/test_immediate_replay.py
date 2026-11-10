"""Offline replay must bind labels without changing the original interaction."""
from copy import deepcopy

import pytest
from test_procredit_rollout import rollout_fixture
from test_simple_turn_scoring import fake_score

from scripts.replay_immediate_credit import by_turn, migrated


def test_review_migration_preserves_source_and_rebinds_every_turn(scratch_dir):
    _, _, record = rollout_fixture(scratch_dir, credit_version='procredit-turn-v4',
                                   final_turn_rule='simple-turn-v1', turn_scorer=fake_score)
    before = deepcopy(record.model_dump())
    project = deepcopy(record.scoring_inputs['reward_project_config'])
    project['project'] = {'reward_version': 'v13-procredit-immediate-independent'}
    project['credit']['final_turn_rule'] = 'simple-turn-v2'
    review = []
    for t, decision in enumerate(record.metadata['simple_turn_labels']['decisions']):
        review.append(dict(turn_index=t, bad_turn=False, correct_refusal=False,
                           reason='Current prefix has no new semantic error.',
                           evidence_message_indices=decision['evidence_message_indices']))
    derived = migrated(record, project, review)
    assert record.model_dump() == before
    assert derived.response_turn_ids == record.response_turn_ids
    assert derived.token_turns == record.token_turns
    assert derived.scoring_inputs['simple_turn_inputs']['version'] == 'simple-turn-v2'
    assert all(not item['revoked_refusals'] for item in
               derived.metadata['simple_turn_labels']['decisions'])


def test_mapping_rejects_observation_credit_and_split_turn_values():
    row = {'response_turn_ids': [0, 0, -1, 1], 'phi': [0, 0, 1]}
    assert by_turn(row, {'token_advantages': [0, 0, 0, .5]}) == [0, .5]
    with pytest.raises(AssertionError):
        by_turn(row, {'token_advantages': [0, 0, .1, .5]})
    with pytest.raises(AssertionError):
        by_turn(row, {'token_advantages': [0, .1, 0, .5]})
