# Immediate progress and independent refusal credit

Approved specification: the user's complete request in this conversation.

Keep ProCredit v4 and trajectory GRPO on R=T+0.5*phi_final. Replace remaining-return
turn centering with 0.5*(phi_after-phi_before), including negative increments. Keep
existing policy/process costs and policy ceiling -1-process_cost. At the final actor
advantage boundary, bad wins (min(A,-1)); every clean correct refusal gets max(A+1,1),
without cross-turn deduplication or retrospective revocation. Truncation zero wins
above all branches. One generation gets at most one refusal bonus.

Keep current main d10a8e7's 35-turn limit, at most two truncation replacements,
DB/COMM/REQ progress-v3, transfer-policy-v2, duplicate queries, Judge topology,
monetary semantic judgment, optimizer and rollout settings. No mandatory calculator,
new Judge stage, numeric matcher, progress floor, repair reward or GPU/API experiment.

Version selectors retain historical behavior; current defaults opt into immediate_delta_v1
and simple-turn-v2. Previous default YAML remains an exact v12 snapshot. New Judge
prompt/cache identity changes only refusal independence instructions, not factual rules.

Implementation choice: branch/PR instead of directly replacing shared main. Offline
sample judgments are AI review, not a new production Judge API run or human gold set.
Controlled replay keeps identical same-task groups, tokens, T/phi and costs; differing
semantic labels are reported separately. Positive delta does NOT guarantee positive net
advantage when the trajectory advantage is negative; implement the approved formula
without adding an unrequested floor.
