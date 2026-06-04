# Reference: Observability Event Catalogue

Load when defining observability/audit (prompt 17). Pair with
`templates/observability_plan_template.md`. Use generic ID patterns; add
domain-specific IDs only when the blueprint justifies them.

## Correlation ID patterns

Generic (use what the system needs):

```text
request_id, trace_id, workflow_id, workflow_step_id, primary_entity_id,
actor_id, decision_id, audit_event_id, external_call_id, artifact_id
```

AI-heavy systems add:

```text
model_call_id, tool_call_id, agent_run_id, review_round_id, evaluation_run_id
```

Domain-specific IDs are examples, not requirements. For a document-translation
system they might become:

```text
job_id, document_id, segment_id, candidate_id, reviewer_id, review_round_id,
domain_profile_id
```

Do not force translation-specific IDs into unrelated projects.

## Log event pattern

```text
<entity>_created; <entity>_validated; <workflow_step>_started;
<workflow_step>_completed; <workflow_step>_failed; <decision>_made;
<escalation>_triggered; <trust_boundary_violation>_detected
```

## Metric families

```text
throughput; latency; error; quality; cost; AI/tool-call;
human-review (where applicable)
```

## Traces (define for)

```text
main end-to-end workflow; critical sub-workflows; AI/agent calls;
tool/MCP calls; external provider calls; human approval flows
```

## Audit requirements

Every final output or state-changing action must be traceable to:

```text
input reference/hash; actor or caller; decision path; intermediate outputs;
validation results; AI/tool calls (if any); human decisions (if any);
final output hash or state change
```

Audit records are append-only; define immutability and retention in §14
(State, Storage, and Data Lifecycle).

## Observability quality gate

> An operator must be able to answer "what happened to this <unit of work>?"
> from the correlation IDs, logs, traces, and audit trail alone.
