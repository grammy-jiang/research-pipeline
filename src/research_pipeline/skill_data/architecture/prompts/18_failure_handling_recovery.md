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

## Evaluator / probe availability policy

If the architecture uses any model-backed evaluator, scorer, or quality probe
(or any external AI dependency that gates a decision), define a single explicit
availability policy — one row per evaluator/probe — so behavior is unambiguous
when it is unavailable or incomplete:

```text
| Probe / Evaluator | Used In | Required Level | If Unavailable | Auto-Accept / Gate-Pass Allowed? | Audit Event |
```

- **Required Level:** optional signal / required signal / release-gate signal /
  diagnostic-only signal.
- **If Unavailable:** fail closed, degrade (mark "unscored"), or escalate to a
  human — state which.
- A **required** or **release-gate** probe that is unavailable must **disable
  auto-accept** of the AI-gated action (e.g. require manual review or fail
  closed). An **optional** probe may degrade. Never auto-accept on an incomplete
  required-signal score.
- Emit an audit event for every unavailable/degraded case.

If the system has no model-backed evaluators, state that explicitly (the gate is
then n/a). This policy is general; the row contents are domain-specific.

## Output

`intermediate/failure_handling.md` → populates §18 (including the
evaluator/probe availability policy when applicable).

## Validation / failure policy

- Gate: no critical workflow lacks failure handling.
- Failure policy: `revise`.
