# Prompt 18 — Failure Handling and Recovery

You are defining how every critical workflow behaves under failure.

## Inputs

- `intermediate/c4_views.md`, `intermediate/observability_audit.md`,
  `intermediate/security_trust_boundaries.md`.
- `rule-packs/reliability_rules.md`.

## Instructions

Define, per the reliability rule pack: timeouts; retry policy (backoff/jitter);
fallback policy; idempotency; partial failure handling; queue/job recovery;
external provider failure behavior; AI model failure behavior; tool/MCP failure
behavior; human approval timeout behavior; data corruption recovery; audit
write failure behavior.

Map each policy to the workflow steps in §15. Define what happens to in-flight
work on crash/restart (resume vs re-run vs dead-letter).

## Output

`intermediate/failure_handling.md` → populates §18.

## Validation / failure policy

- Gate: no critical workflow lacks failure handling.
- Failure policy: `revise`.
