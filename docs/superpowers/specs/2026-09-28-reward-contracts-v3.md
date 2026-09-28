# Reward contracts v3

User-approved scope: fix the eight issues audited after ac96868; preserve terminal
policy gates, keep original rollouts auditable, and explicitly use BF16 (user reply).
Implementation is authorized by “请继续修改”; no additional design approval is needed.

## Decisions

1. Canonical completion matching follows the pinned Airline tool inputs. Flight,
   passenger and payment objects compare only fields consumed by their data models.
   Flight order, target IDs, actual monetary amounts and meaningful values remain
   significant. Numeric normalization is restricted to numeric fields.
2. Safety is a separate write-scope predicate: allowed tool and reservation/user
   identity, rather than equality to the final reference payload. Policy Judge
   continues to enforce consent, eligibility and payment rules; completion/DB checks
   still determine task success. Append-only bookings/certificates cannot exceed
   the annotated number for the same write scope.
3. Capture the pre-call baggage count per executed tool, inside the existing audit
   hook. A different payment ID is completion-equivalent only when the update is
   proven to charge zero; missing state is conservative (the existing zero-target
   case remains compatible with the nonnegative baggage invariant).
4. Policy checks have an explicit applicability field. Absent legacy fields mean
   applicable, preserving historical interpretation. New Judge requests require it.
   N/A must pass and carry no blame. Concrete tool-triggered rules cannot be
   bypassed by a Judge N/A verdict. Prompt/rubric/cache versions change together.
   The transfer execution-fact fix remains in place.
5. v3 keeps scalar S = V * max(0, m*(R+.5*Phi)-P). Its turn credit additionally
   subtracts capped, deduplicated per-turn process costs after centering. Costs
   remain active when S is zero and never propagate to earlier turns. If the same
   turn already has a policy penalty, use the stronger local penalty rather than
   double-charging the same turn. lambda_turn=0 disables all local credit.
6. Progress v2 uses one state-goal family: with an applicable DB verifier, only DB
   and COMM count towards Phi; required-action checks remain audited but have zero
   progress weight. Without DB, REQ can supply progress. Initially met checks have
   zero weight. DB regression remains observable. New definitions are fingerprinted.
7. For v3 training, filter the approved train/smoke parquet rows using an initial
   prefix progress check in a fresh, non-LLM Tau2 evaluator. Save reasons/identities
   and use a separate deterministic dataset artifact. Never promote dev/test tasks
   into training or filter their evaluation rows. Fail early if no train tasks
   remain. No per-prefix Judge calls or fabricated reasoning progress.
8. Explicit BF16 loading, mixed-precision parameters and rollout; FP32 reductions
   remain allowed. Reject overrides that silently diverge actor and rollout.
   Record the settings in launch/resume identity.

## Version and acceptance boundaries

Use separate airline_procredit_v3.yaml / procredit-turn-v3. v1/v2 pure group-credit
replay and serialized configs remain compatible; newly scored data binds updated
matcher/Judge/progress/process contracts. Old runs cannot resume as v3.

Acceptance covers equivalent-flight and zero-charge baggage cases; safety-scope
violations and intermediate writes; N/A versus actual tool triggers; zero-score
process errors; no duplicated state-goal weight; signal eligibility without data
leakage; BF16 launcher wiring; original group membership and policy-token masks.
No real API/GPU training is automatically started. Local tests do not prove
on-policy smoke success or long-run policy compliance.
