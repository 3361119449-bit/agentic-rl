# Reward contracts v3 implementation

Spec: ../specs/2026-09-28-reward-contracts-v3.md
Base: ac96868. Execute inline, test-first, with one independent review at the end.
Workspace: isolated codex/reward-contracts-v3.

1. Matchers and baggage state
   - Files: reward/required_actions.py, reward/mandatory_policy.py, schemas.py,
     environment/tau2_gym.py, agent_loop/airline.py.
   - Interface: canonical tool comparison; separate matches_allowed_write scope;
     optional ToolEvent.state_before, captured per call and frozen in records.
   - Red/green cases: extra FlightInfo fields; preserved flight order and IDs;
     same-scope intermediate writes vs completion; excess creation; zero-charge
     paid-bag update versus real charge/missing pre-state; original tests.
2. Judge applicability
   - Files: schemas.py, policy_rules.py, judge/{client,prompts,evidence}.py,
     reward/{score,policy_credit}.py, policy annotation JSONs.
   - Require explicit applicability for new responses; preserve legacy model
     defaults. Normalize/validate hard trigger facts and reject contradictory N/A.
   - Red/green: inactive booking rules, active writes cannot bypass consent,
     missing field, inconsistent blame, transfer normalization, cached responses.
3. Progress and process credit
   - Files: reward/{progress,score,process_penalty}.py, advantages.py,
     procredit_runtime.py, actor, scoring retry, offline scoring.
   - New trace version/weights; immutable local process trace built from events
     and configured caps, including over-limit turns. Actor/replay share inputs.
   - Red/green: DB+REQ not doubled, DB regressions, REQ-only task, zero-score
     process error survives filtering and tensors; no observation penalties;
     caps/deduplication and legacy audit replay.
4. Training eligibility and BF16
   - Files: new training-selection helper, train_airline_grpo.py,
     training_config.py, config.py, airline_procredit_v3.yaml.
   - Initial-prefix evaluator supplied from pinned Tau2, no model calls; approved
     train subset only; separately content-bound parquet and eligibility manifest.
   - Red/green: no-op exclusion, signal-bearing inclusion, held-out data untouched,
     empty selection refusal, BF16 wiring/override conflicts and resume identity.
5. Integration, documentation and validation
   - Tests: production actor -> reward -> queue -> trainer and frozen retry;
     final full CPU/Torch/TensorDict suite, Ruff, whitespace check.
   - Re-evaluate relevant original smoke matcher evidence offline without
     injecting historical samples into the online queue.
   - Fresh read-only reviewer; fix material findings with regression tests.
   - Document exact limitations, new config/launch instructions and results.
   - Commit and push the dedicated branch using earlier user authorization.

Review focus: unsafe normalization of IDs/amounts; per-call state in multitool
batches; Judge N/A hiding actual violations; clipped cost on empty-token turns;
training eligibility or dtype changed without changing resume identity.
