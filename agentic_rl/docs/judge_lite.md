# Judge Lite

This revision keeps the original global Judge and one per-turn Judge call for each
assistant generation. It adds no conflict Judge, transfer-facts Judge, required audit
categories, monetary output schema or reward orchestration. Normal flat output fields
and target-v2 identity/evidence binding are unchanged.

The compact per-turn prompt covers prices/counts/net refunds, latest charge consent,
wrong dates/airports, unsupported facts, policy exceptions and factual transfer
summaries. The global prompt asks for independent blame rather than propagating earlier
errors or treating reference actions as policy. These are guidance changes; offline
checks alone cannot establish improved semantic accuracy.

A small local response check rejects explicit error/neutral contradictions and an
error simultaneously labelled as a clean refusal. It never turns a keyword into a new
label. The existing bounded transport loop supplies the validation error to the same
frozen request; no extra review stage or retry loop is added. Quoted and negated error
statements are excluded from the narrow contradiction check. Old frozen record schema
validation and reward/advantage equations remain unchanged.

Cache identity includes updated prompt versions. Cache directories and target hashes
remain unchanged. The logical prompt hash addresses the original request; cached
accepted_request_sha256 records the actual accepted retry payload, and validation_history
keeps rejected raw outputs with their request hashes for audit.

Development validation is static/offline: socket-blocked fixtures and saved response
inspection. No automatic online acceptance, rollout or GPU training is part of this
change. API decoding,35-turn limits, cutoff/resampling rules, concurrency8, BF16 and
full tool observation settings are unchanged. Existing deterministic reference-action
matching rules are also unchanged.
