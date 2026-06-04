# Prompt 17 — Observability, Logging, Telemetry, and Audit

You are defining how the running system is observed and audited.

## Inputs

- `intermediate/c4_views.md`, `intermediate/interface_contracts.md`,
  `intermediate/security_trust_boundaries.md`.
- `templates/observability_plan_template.md`,
  `references/observability_event_catalogue.md`.

## Instructions

Produce all of §16:

1. **Correlation IDs** — pick from the generic pattern (request_id, trace_id,
   workflow_id, workflow_step_id, primary_entity_id, actor_id, decision_id,
   audit_event_id, external_call_id, artifact_id) and, for AI-heavy systems,
   add model_call_id, tool_call_id, agent_run_id, review_round_id,
   evaluation_run_id. Use domain-specific IDs only when the blueprint justifies
   them.
2. **Logs** — define events for every critical workflow using the
   `<entity>_/<step>_/<decision>_` pattern.
3. **Metrics** — throughput, latency, error, quality, cost, AI/tool-call, and
   human-review (where applicable).
4. **Traces** — main workflow, critical sub-workflows, AI/agent calls,
   tool/MCP calls, external provider calls, human approval flows.
5. **Audit** — every final output or state-changing action is traceable to
   input reference/hash, actor, decision path, intermediate outputs, validation
   results, AI/tool calls, human decisions, and final output hash/state change.

## Output

`intermediate/observability_audit.md` → populates §16.

## Validation / failure policy

- Gate: correlation IDs, logs, metrics, traces, and the audit trail are all
  present; the "what happened to this unit of work?" test passes.
- Failure policy: `revise`.
