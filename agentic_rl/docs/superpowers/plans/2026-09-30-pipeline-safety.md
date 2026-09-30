# RL pipeline safety implementation plan

> Execute inline with systematic debugging, TDD, and verification before completion.

Goal: preserve useful groups after transient exhaustion, stop on fatal failures, and make precision and reward defaults explicit.

1. Launcher defaults (`scripts/train_airline_grpo.py`, `scripts/evaluate_airline.py`, evaluation YAML, README): default training to ProCredit v4, evaluation to official plus custom strict, and explicitly load/compute BF16 for every evaluation split. Test generated commands and the explicit official-only switch.
2. Actor accounting (`training_config.py`, `verl_capped_trainer.py`): GRPO has no critic warmup; reject nonzero warmup before rollout, including runtime and parent/deletion overrides. Test rejection and zero acceptance.
3. Failure propagation (`concurrency.py`, a pinned veRL worker adapter, `verl_entrypoint.py`, `verl_capped_trainer.py`): preserve a job-wide first-fatal latch before veRL collapses session exceptions; check before sampling/refill/actor updates. Transient exhaustion still quarantines only its group. Test a fatal sibling alongside good groups and pending siblings.
4. Frozen scoring (`judge/client.py`, `scoring_retry.py`, agent loop): explicitly distinguish retryable Judge response/service failures from deterministic code/config errors; unknown errors must not become ordinary exhaustion. Test fatal scoring and transport retry separately.
5. Run targeted regressions, the complete local suite, Ruff, and diff checks. Review the resulting diff. Do not push without a new request.

Ruling: reject critic warmup rather than simulate critic-only steps because this pipeline uses GRPO without a critic; optimizer and policy version counters remain actor-update counters.
