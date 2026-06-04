# Observability Plan Template

Cover correlation IDs, logs, metrics, traces, and the audit trail. Use the
generic correlation-ID pattern and only add domain-specific IDs that the
blueprint justifies.

## 16.1 Correlation IDs

Generic pattern (use what the system needs; do not force unused IDs):

```text
request_id, trace_id, workflow_id, workflow_step_id, primary_entity_id,
actor_id, decision_id, audit_event_id, external_call_id, artifact_id
```

For AI-heavy systems add:

```text
model_call_id, tool_call_id, agent_run_id, review_round_id, evaluation_run_id
```

| ID | Scope | Propagated to | Notes |
|---|---|---|---|
| `workflow_id` | one end-to-end run | logs, traces, audit | <…> |
| `model_call_id` | one LLM call | logs, traces, cost metrics | <…> |

## 16.2 Logs

Use a structured event pattern:

```text
<entity>_created, <entity>_validated, <workflow_step>_started,
<workflow_step>_completed, <workflow_step>_failed, <decision>_made,
<escalation>_triggered, <trust_boundary_violation>_detected
```

| Event | Level | Fields | Emitted by |
|---|---|---|---|
| `<step>_completed` | INFO | workflow_id, duration_ms | <module> |

## 16.3 Metrics

| Metric | Type | Labels | Why |
|---|---|---|---|
| throughput | counter | workflow | capacity |
| latency | histogram | step | SLOs |
| errors | counter | category | reliability |
| quality | gauge/histogram | <…> | output quality |
| cost | counter | model | spend |
| ai_tool_calls | counter | tool, outcome | AI usage/safety |
| human_review | counter | outcome | HITL load |

## 16.4 Traces

Define traces for: the main end-to-end workflow, critical sub-workflows,
AI/agent calls, tool/MCP calls, external provider calls, and human approval
flows.

## 16.5 Audit Trail

Every final output or state-changing action must be traceable to:

```text
input reference/hash, actor or caller, decision path, intermediate outputs,
validation results, AI/tool calls (if any), human decisions (if any),
final output hash or state change
```

Audit records are append-only; define immutability and retention in §14.
