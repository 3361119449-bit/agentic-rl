from copy import deepcopy

import pytest
from test_procredit_credit import row

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit


@pytest.mark.parametrize('a0,bad,good,want', [
    (2.802491, True, False, -1.0),
    (-4.557508, True, False, -4.557508),
    (-0.2, True, False, -1.0),
    (-0.628714799, False, True, 1.0),
    (0.0, False, True, 1.0),
    (1.930664945, False, True, 2.930664945),
    (3.533402572, True, True, -1.0),
    (2.882490619, False, False, 2.882490619),
    (-0.539868, False, False, -0.539868),
])
def test_report_numeric_cases(a0, bad, good, want):
    from tau2_agentic_rl.reward.simple_turn import transform_advantage

    assert transform_advantage(a0, bad, good) == pytest.approx(want)


def v4_row(index=0):
    return row(index, credit_version='procredit-turn-v4',
               policy_credit={'version': 'policy-credit-v1', 'violating_turns': [],
                              'attribution_complete': True, 'unresolved_checks': []},
               process_credit={'version': 'process-credit-v2', 'turn_costs': [0, 0]})


def test_group_uses_labels_only_after_original_centering_and_masks_observations():
    rows = [v4_row(i) for i in range(8)]
    original = compute_group_credit(rows, CreditConfig(version='procredit-turn-v4'))
    for r in rows:
        r['simple_turn_labels'] = {'version': 'simple-turn-v1', 'turns': [
            {'turn_index': 0, 'bad_turn': True, 'eligible_correct_refusal': False, 'evidence': []},
            {'turn_index': 1, 'bad_turn': False, 'eligible_correct_refusal': True, 'evidence': []},
        ]}
    snapshot = deepcopy(rows)
    config = CreditConfig(version='procredit-turn-v4', final_turn_rule='simple-turn-v1')
    result = compute_group_credit(rows, config)
    assert rows == snapshot
    assert result['score_mean'] == original['score_mean'] == 1.5
    assert result['turn_mean'] == original['turn_mean'] == 1.25
    actual = result['trajectories'][0]
    assert actual['token_advantages'] == [-1, -1, 0, 1, 1]
    assert actual['simple_turn_audit'][0]['a0'] == 0.25
    assert actual['simple_turn_audit'][1]['a0'] == -0.25


def test_enabled_rule_cannot_silently_accept_unlabeled_rows():
    config = CreditConfig(version='procredit-turn-v4', final_turn_rule='simple-turn-v1')
    with pytest.raises(ValueError, match='label'):
        compute_group_credit([v4_row()], config)


def test_legacy_config_fingerprint_and_results_remain_unchanged():
    config = CreditConfig(version='procredit-turn-v4')
    result = compute_group_credit([v4_row()], config)
    assert 'final_turn_rule' not in config.audit_config()
    assert 'simple_turn_audit' not in result['trajectories'][0]
    assert result['trajectories'][0]['token_advantages'] == [0.25, 0.25, 0, -0.25, -0.25]


def decision(turn, *, bad=False, refusal=False, operation='', target='', revoked=None):
    return {'turn_index': turn, 'bad_turn': bad, 'correct_refusal': refusal,
            'reason': 'Policy forbids changing passenger count.' if refusal else 'Reviewed visible evidence.',
            'evidence_message_indices': [2 * turn + 1] if revoked else [0],
            'refusal_requests': [{'target': target, 'operation': operation}] if refusal else [],
            'revoked_refusals': [{'turn_index': t, 'target': obj, 'operation': op}
                                 for t, obj, op in (revoked or [])]}


def inputs(turns=3):
    from tau2_agentic_rl.reward.simple_turn import build_turn_inputs

    messages = [{'role': 'user', 'content': 'Remove one passenger from booking A.'}]
    for t in range(turns):
        messages.append({'role': 'assistant', 'turn_idx': t + 1, 'content': f'reply {t}'})
        messages.append({'role': 'user', 'content': f'next request {t}'})
    return build_turn_inputs(messages, 'Passenger count cannot be changed.', turns)


def trace(decisions, events=(), bad=()):
    from tau2_agentic_rl.reward.simple_turn import build_label_trace

    return build_label_trace(inputs(len(decisions)), decisions, list(events),
                            {'violating_turns': list(bad), 'attribution_complete': True})


def test_refusal_dedup_uses_target_and_operation_and_bad_wins():
    result = trace([
        decision(0, refusal=True, target='A', operation='remove_passenger'),
        decision(1, refusal=True, target='A', operation='remove_passenger'),
        decision(2, bad=True, refusal=True, target='A', operation='change_cabin'),
        decision(3, refusal=True, target='A', operation='cancel'),
    ])
    assert [t['eligible_correct_refusal'] for t in result['turns']] == [True, False, False, True]
    assert result['turns'][2]['bad_turn'] is True


def test_later_illegal_reversal_only_revokes_same_request_without_backdated_bad():
    result = trace([
        decision(0, refusal=True, target='A', operation='remove_passenger'),
        decision(1, refusal=True, target='A', operation='cancel'),
        decision(2, bad=True, revoked=[(0, 'A', 'remove_passenger')]),
    ])
    assert [t['eligible_correct_refusal'] for t in result['turns']] == [False, True, False]
    assert [t['bad_turn'] for t in result['turns']] == [False, False, True]


def test_unrelated_bad_transfer_does_not_cancel_refusal():
    result = trace([decision(0, refusal=True, target='A', operation='cancel'), decision(1, bad=True)])
    assert result['turns'][0]['eligible_correct_refusal'] is True


def test_tool_errors_are_whole_turn_and_empty_results_done_or_repair_are_not_bad():
    from tau2_agentic_rl.schemas import ToolEvent

    events = [
        ToolEvent(event_id='ok', sequence=0, turn_id=1, success=True),
        ToolEvent(event_id='bad', sequence=1, turn_id=1, error_kind='model_caused_execution_error'),
        ToolEvent(event_id='empty', sequence=2, turn_id=2, success=True, result='[]'),
        ToolEvent(event_id='stop', sequence=3, turn_id=3, execution_status='termination_control'),
    ]
    result = trace([decision(0), decision(1), decision(2)], events)
    assert [t['bad_turn'] for t in result['turns']] == [True, False, False]


def test_old_policy_bad_is_preserved_without_new_semantic_labels():
    result = trace([decision(0), decision(1)], bad=[1])
    assert [t['bad_turn'] for t in result['turns']] == [False, True]


def test_semantic_evidence_cannot_reference_future_or_hidden_messages():
    d = decision(0, bad=True)
    d['evidence_message_indices'] = [2]
    with pytest.raises(ValueError, match='evidence'):
        trace([d, decision(1)])


def test_only_true_generation_messages_count_and_neutral_prefix_has_no_future():
    from tau2_agentic_rl.reward.simple_turn import build_turn_inputs, turn_request

    frozen = build_turn_inputs([
        {'role': 'system', 'content': 'policy'},
        {'role': 'assistant', 'content': 'initial scripted greeting'},
        {'role': 'user', 'content': 'request'},
        {'role': 'assistant', 'turn_idx': 1, 'content': 'generated'},
        {'role': 'tool', 'content': 'future secret'},
        {'role': 'assistant', 'turn_idx': 2, 'content': 'later generation'},
    ], 'bound policy', 2)
    request = turn_request(frozen, 0, [])
    assert len(request['messages']) == 4
    assert request['messages'][-1]['message']['content'] == 'generated'
    assert 'future secret' not in str(request)


def test_revocation_requires_current_generation_evidence():
    d = decision(1, bad=True, revoked=[(0, 'A', 'cancel')])
    d['evidence_message_indices'] = [0]
    with pytest.raises(ValueError, match='current'):
        trace([decision(0, refusal=True, target='A', operation='cancel'), d])


def test_queue_cannot_override_old_policy_bad_with_a_refusal_label():
    r = v4_row()
    r.update(valid=False)
    r['policy_credit']['violating_turns'] = [0]
    r['simple_turn_labels'] = trace([decision(0, refusal=True, target='A', operation='cancel'), decision(1)])
    with pytest.raises(ValueError, match='policy'):
        compute_group_credit([r], CreditConfig(version='procredit-turn-v4', final_turn_rule='simple-turn-v1'))


@pytest.mark.parametrize('source', ['policy', 'tool'])
def test_revocation_via_bad_repeated_refusal_still_revokes_first_award(source):
    from tau2_agentic_rl.schemas import ToolEvent

    events = [ToolEvent(event_id='bad', sequence=0, turn_id=2, error_kind='parse_error')]
    result = trace([
        decision(0, refusal=True, target='A', operation='cancel'),
        decision(1, refusal=True, target='A', operation='cancel'),
        decision(2, bad=True, revoked=[(1, 'A', 'cancel')]),
    ], events=events if source == 'tool' else [], bad=[1] if source == 'policy' else [])
    assert [t['eligible_correct_refusal'] for t in result['turns']] == [False, False, False]
    assert result['turns'][0]['revoked_requests'] == [{'dedup_key': ['a', 'cancel'], 'revoked_by_turn': 2}]
    assert result['turns'][0]['bad_turn'] is False
    assert result['turns'][1]['bad_turn'] is True


def test_same_turn_multiple_refusals_register_both_keys_but_award_once():
    d = decision(0, refusal=True, target='A', operation='remove_passenger')
    d['refusal_requests'] = [
        {'target': 'A', 'operation': 'remove_passenger'},
        {'target': 'A', 'operation': 'cancel'},
    ]
    result = trace([d, decision(1, refusal=True, target='A', operation='cancel')])
    assert [t['eligible_correct_refusal'] for t in result['turns']] == [True, False]
    assert result['turns'][0]['refusal_bonus'] == 1


@pytest.mark.parametrize('operations,want', [(['cancel'], True), (['cancel', 'remove_passenger'], False)])
def test_reversal_of_one_of_multiple_refusals_only_revokes_that_request(operations, want):
    d = decision(0, refusal=True, target='A', operation='remove_passenger')
    d['refusal_requests'] = [
        {'target': 'A', 'operation': 'remove_passenger'},
        {'target': 'A', 'operation': 'cancel'},
    ]
    reverse = decision(1, bad=True)
    reverse['evidence_message_indices'] = [3]
    reverse['revoked_refusals'] = [
        {'turn_index': 0, 'target': 'A', 'operation': operation} for operation in operations
    ]
    result = trace([d, reverse])
    assert result['turns'][0]['eligible_correct_refusal'] is want
    assert result['turns'][0]['bad_turn'] is False


def test_revocation_must_identify_a_request_present_in_the_cited_turn():
    with pytest.raises(ValueError, match='revocation'):
        trace([decision(0, refusal=True, target='A', operation='remove_passenger'),
               decision(1, bad=True, revoked=[(0, 'A', 'cancel')])])


def test_bad_refusal_does_not_consume_first_valid_refusal_award():
    result = trace([decision(0, refusal=True, target='A', operation='cancel'),
                    decision(1, refusal=True, target='A', operation='cancel')], bad=[0])
    assert [t['eligible_correct_refusal'] for t in result['turns']] == [False, True]
