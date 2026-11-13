"""Offline credit comparison on a frozen reward bundle; never calls a model.

Historical decisions are explicitly reused (not newly validated Judge predictions).
Optional review labels are replayed twice: first with original policy attribution,
then with explicitly reviewed policy attribution. The latter is a labeled numerical
counterfactual, not a newly scored production record. Native tool evidence, task
completion, progress, groups and process costs are held fixed in both comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.procredit_runtime import credit_from_record
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.reward.simple_turn import (
    build_label_trace,
    build_turn_inputs,
    turn_request,
)
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.versions import sha256_json


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def score(record, project):
    x = record.scoring_inputs
    return score_trajectory(
        config=build_reward_config(project), progress_trace=record.progress_trace,
        events=record.tool_events, messages=record.environment_transcript,
        assistant_turns=record.assistant_turns, official=record.official_scores,
        judge=record.judge_result, required_actions=x['required_actions'],
        transfer_rule=x['judge']['transfer_rule'], action_dependencies=x['action_dependencies'],
        termination_reason=record.termination_reason,
    )


def migrated(record, project, review=None):
    """New derived record; source inputs/labels remain untouched on disk."""
    new = record.model_copy(deep=True)
    new.reward_version = project['project']['reward_version']
    new.metadata['offline_credit_replay'] = {
        'source_reward_version': record.reward_version,
        'source_scoring_inputs_sha256': record.metadata['scoring_inputs_sha256'],
        'label_source': 'AI current-prefix review' if review else 'frozen decisions, no new Judge call',
    }
    new.scoring_inputs['reward_project_config'] = {
        k: deepcopy(project[k]) for k in ('reward', 'credit', 'rollout')}
    new.custom_reward = score(new, project)
    rule = project['credit']['final_turn_rule']
    x = build_turn_inputs(new.messages, new.scoring_inputs['judge']['policy'],
                          new.assistant_turns, version=rule)
    new.scoring_inputs['simple_turn_inputs'] = x
    decisions = deepcopy(record.metadata['simple_turn_labels']['decisions'])
    if rule == 'simple-turn-v2':
        for d in decisions:
            d['revoked_refusals'] = []  # Explicit migration, not a new model inference.
    if review is not None:
        assert len(review) == new.assistant_turns
        for t, (d, label) in enumerate(zip(decisions, review, strict=True)):
            assert label['turn_index'] == t
            req = turn_request(x, t, [])
            d.update(bad_turn=label['bad_turn'], correct_refusal=label['correct_refusal'],
                     reason=label['reason'], evidence_message_indices=label['evidence_message_indices'],
                     refusal_requests=label.get('refusal_requests', []), revoked_refusals=[],
                     target_quote=req['target_generation_text'][:160])
    if not new.custom_reward.details.get('truncation_zero_applied'):
        new.metadata['simple_turn_labels'] = build_label_trace(
            x, decisions, new.tool_events, new.custom_reward.details['policy_credit'])
    else:
        # The production cutoff path skips Judge and labels. Retain raw review
        # externally, never let a refusal/error label revive a truncated sample.
        new.metadata.pop('simple_turn_labels', None)
    new.metadata['scoring_inputs_sha256'] = sha256_json(new.scoring_inputs)
    return new


def by_turn(row, credit):
    mapping = row['response_turn_ids']
    tokens = credit['token_advantages']
    assert len(tokens) == len(mapping)
    groups = defaultdict(set)
    for t, value in zip(mapping, tokens, strict=True):
        if t < 0:
            assert value == 0
        else:
            groups[t].add(value)
    assert set(groups) == set(range(len(row['phi']) - 1))
    assert all(len(v) == 1 for v in groups.values())
    return [next(iter(groups[t])) for t in range(len(groups))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--review-labels', type=Path)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty; never overwrite a prior audit')
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    before = load_yaml(root / 'configs/rl/airline_procredit_v4_v12.yaml')
    after = load_yaml(root / 'configs/rl/airline_procredit_v4.yaml')
    review = json.loads(args.review_labels.read_text()) if args.review_labels else {}
    originals, versions, grouped, sources = {}, {}, defaultdict(list), []
    for path in sorted((args.bundle_root / 'trajectories').glob('*.json')):
        data = path.read_bytes()
        r = TrajectoryRecord.model_validate_json(data)
        assert score(r, r.scoring_inputs['reward_project_config']) == r.custom_reward
        raw_row = credit_from_record(r)
        original_hash = sha256_json(r.model_dump())
        v12 = migrated(r, before)
        v13 = migrated(r, after)
        checked = migrated(r, after, review[r.trajectory_id]) if r.trajectory_id in review else v13
        rows = [raw_row, credit_from_record(v12), credit_from_record(v13), credit_from_record(checked)]
        reviewed_policy_row = deepcopy(rows[3])
        if r.trajectory_id in review and rows[3]['multiplier'] != 0:
            bad = [i for i, label in enumerate(review[r.trajectory_id]) if label['policy_violation']]
            reviewed_policy_row['policy_credit'] = {
                'version': 'policy-credit-v1', 'violating_turns': bad,
                'attribution_complete': True, 'unresolved_checks': [],
            }
            reviewed_policy_row['valid'] = not bad
            reviewed_policy_row['simple_turn_labels'] = build_label_trace(
                checked.scoring_inputs['simple_turn_inputs'],
                checked.metadata['simple_turn_labels']['decisions'], checked.tool_events,
                reviewed_policy_row['policy_credit'])
        rows.append(reviewed_policy_row)
        assert all(x['response_turn_ids'] == raw_row['response_turn_ids'] for x in rows)
        assert rows[1]['score'] == rows[2]['score'] == rows[3]['score']
        assert rows[1]['phi'] == rows[2]['phi'] == rows[3]['phi']
        assert rows[1]['process_credit'] == rows[2]['process_credit'] == rows[3]['process_credit']
        assert rows[1]['policy_credit'] == rows[2]['policy_credit'] == rows[3]['policy_credit']
        assert sha256_json(r.model_dump()) == original_hash
        originals[r.trajectory_id] = r
        versions[r.trajectory_id] = rows
        grouped[r.task_id].append(r.trajectory_id)
        sources.append({'file': str(path.relative_to(args.bundle_root)),
                        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    assert len(originals) == 80 and len(grouped) == 20
    turns, trajs = [], []
    for task, ids in sorted(grouped.items(), key=lambda x: int(x[0])):
        ids.sort(key=lambda key: originals[key].metadata['evaluation_sample_index'])
        assert len(ids) == 4
        archive = json.loads((args.bundle_root / 'exports/group_audits' / f'task_{task}.json').read_text())
        configs = [CreditConfig.from_project(originals[ids[0]].scoring_inputs['reward_project_config']),
                   CreditConfig.from_project(before), CreditConfig.from_project(after),
                   CreditConfig.from_project(after), CreditConfig.from_project(after)]
        results = [compute_group_credit([versions[k][i] for k in ids], cfg)
                   for i, cfg in enumerate(configs)]
        archived = {v['trajectory_id']: v for v in archive['new_credit']['trajectories']}
        for j, key in enumerate(ids):
            record = originals[key]
            rows = versions[key]
            reports = [r['trajectories'][j] for r in results]
            assert reports[0]['token_advantages'] == archived[key]['token_advantages']
            values = [by_turn(row, result) for row, result in zip(rows, reports, strict=True)]
            badset = set(rows[2]['policy_credit']['violating_turns'])
            labels = rows[2].get('simple_turn_labels', {}).get('turns', [{}] * record.assistant_turns)
            checked_labels = rows[4].get('simple_turn_labels', {}).get('turns', [{}] * record.assistant_turns)
            token_counts = Counter(rows[2]['response_turn_ids'])
            group_mean = results[2]['score_mean']
            group_std = results[2]['score_std']
            B = reports[2]['trajectory_advantage']
            source_indices = record.scoring_inputs['simple_turn_inputs']['turn_message_indices']
            for t in range(record.assistant_turns):
                delta = rows[2]['phi'][t + 1] - rows[2]['phi'][t]
                pp = float(t in badset)
                pc = rows[2]['process_credit']['turn_costs'][t]
                raw = B + .5 * delta - pp - pc
                policy_guard = min(raw, -1 - pc) if t in badset else raw
                label = labels[t]
                independent = (min(policy_guard, -1) if label.get('bad_turn') else
                               max(policy_guard + 1, 1) if label.get('eligible_correct_refusal') else policy_guard)
                if rows[2]['multiplier'] == 0:
                    independent = raw = policy_guard = 0.
                assert math.isclose(independent, values[2][t], rel_tol=0, abs_tol=1e-12)
                for ix, ls in ((2, labels), (4, checked_labels)):
                    if rows[ix]['multiplier'] == 0:
                        assert values[ix][t] == 0
                    elif ls[t]['bad_turn']:
                        assert values[ix][t] <= -1
                    elif ls[t]['eligible_correct_refusal']:
                        assert values[ix][t] >= 1
                # Independently check the AI-policy counterfactual too; its
                # labels are never substituted into the frozen-label comparison.
                checked_raw = B + .5 * delta - float(t in rows[4]['policy_credit']['violating_turns']) - pc
                checked_guard = (min(checked_raw, -1 - pc)
                                 if t in rows[4]['policy_credit']['violating_turns'] else checked_raw)
                checked_expected = (min(checked_guard, -1) if checked_labels[t].get('bad_turn') else
                                    max(checked_guard + 1, 1) if checked_labels[t].get('eligible_correct_refusal')
                                    else checked_guard)
                if rows[4]['multiplier'] == 0:
                    checked_expected = 0.
                assert math.isclose(checked_expected, values[4][t], rel_tol=0, abs_tol=1e-12)
                manual = review.get(key, [None] * record.assistant_turns)[t]
                turns.append(dict(
                    task_id=int(task), sample=record.metadata['evaluation_sample_index'], trajectory_id=key,
                    assistant_turn=t + 1, message_index=source_indices[t], actor_tokens=token_counts[t],
                    termination_reason=record.termination_reason, zero_cutoff=rows[2]['multiplier'] == 0,
                    task_completion=rows[2]['terminal_success'], score=rows[2]['score'],
                    group_mean=group_mean, group_std=group_std, trajectory_advantage=B,
                    phi_before=rows[2]['phi'][t], phi_after=rows[2]['phi'][t + 1],
                    progress_delta=delta, local_progress=.5 * delta, policy_cost=pp, process_cost=pc,
                    base_a0=raw, after_policy_ceiling=policy_guard,
                    v11_archived_advantage=values[0][t], v12_current_baseline_advantage=values[1][t],
                    v13_frozen_decisions_advantage=values[2][t],
                    saved_semantic_bad=record.metadata['simple_turn_labels']['decisions'][t]['bad_turn'],
                    final_bad=label.get('bad_turn', False),
                    independent_refusal=label.get('eligible_correct_refusal', False),
                    review_bad=manual['bad_turn'] if manual else '',
                    review_correct_refusal=manual['correct_refusal'] if manual else '',
                    reviewed_final_bad=checked_labels[t].get('bad_turn', False) if manual else '',
                    reviewed_final_refusal=checked_labels[t].get('eligible_correct_refusal', False) if manual else '',
                    local_review_original_policy_advantage=values[3][t] if manual else '',
                    reviewed_policy_cost=float(t in rows[4]['policy_credit']['violating_turns']) if manual else '',
                    reviewed_advantage=values[4][t] if manual else '',
                    review_reason=manual['reason'] if manual else '',
                    evidence_indices=json.dumps(manual['evidence_message_indices']) if manual else '',
                ))
            trajs.append(dict(task_id=int(task), sample=record.metadata['evaluation_sample_index'],
                              trajectory_id=key, turns=record.assistant_turns,
                              official=record.official_scores.reward,
                              R=rows[2]['score'], T=rows[2]['terminal_success'],
                              phi_final=rows[2]['phi'][-1], raw_phi_final=rows[0]['phi'][-1],
                              B=B, truncated=rows[2]['multiplier'] == 0, reviewed=key in review))
    assert len(turns) == 1227
    for name, data in [('all_1227_turns.csv', turns), ('trajectories_80.csv', trajs),
                       ('reviewed_turns.csv', [x for x in turns if x['review_reason']])]:
        with (out / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]) if data else list(turns[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = dict(
        scope='Offline arithmetic and AI semantic review; no generation, training or Judge API calls',
        archived_baseline='v11, source file', current_baseline='d10a8e7 v12, saved Judge decisions reused',
        group_size=4, online_group_size=after['rollout']['group_size'],
        source_records=80, turns=len(turns), actor_tokens=sum(x['actor_tokens'] for x in turns),
        reviewed_trajectories=len(review), reviewed_turns=sum(bool(x['review_reason']) for x in turns),
        truncated_trajectories=sum(x['truncated'] for x in trajs),
        zeroed_turns=sum(x['zero_cutoff'] for x in turns),
        bad_turns=sum(x['final_bad'] for x in turns),
        rewarded_refusal_turns=sum(x['independent_refusal'] for x in turns),
        old_refusals=sum(label['eligible_correct_refusal'] for rows in versions.values()
                         for label in rows[1].get('simple_turn_labels', {}).get('turns', [])),
        changed_turns=sum(not math.isclose(x['v12_current_baseline_advantage'],
                                          x['v13_frozen_decisions_advantage'], abs_tol=1e-12)
                          for x in turns),
        labels_reused_not_new_judge=True, source_hashes_verified=True,
        original_v11_reproduction=True, independent_delta_formula_assertions=True,
        token_mapping_and_masks_verified=True, zero_cutoff_assertions=True,
        policy_and_process_unchanged=True, scalar_between_v12_v13_unchanged=True,
        note='Both local-only review (original policy retained) and full AI-reviewed policy counterfactual are exported; neither is an API call.',
    )
    for item in sources:
        raw = (args.bundle_root / item['file']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    dump(out / 'summary.json', summary)
    dump(out / 'source_manifest.json', sources)
    dump(out / 'current_baseline_config.json', before)
    dump(out / 'new_config.json', after)
    if review:
        dump(out / 'review_labels.json', review)
        contexts = {key: {'task_id': r.task_id, 'sample': r.metadata['evaluation_sample_index'],
                         'messages': r.messages, 'policy': r.scoring_inputs['judge']['policy'],
                         'tool_events': [e.model_dump() for e in r.tool_events]}
                    for key, r in originals.items() if key in review}
        dump(out / 'reviewed_source_contexts.json', contexts)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
