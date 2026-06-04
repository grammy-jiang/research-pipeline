# Checklist: Observability

- [ ] Correlation IDs defined (generic pattern; AI IDs where AI-heavy;
      domain-specific IDs only when justified).
- [ ] Log events defined for every critical workflow.
- [ ] Metrics cover throughput, latency, errors, and cost at minimum (plus
      quality, AI/tool-call, human-review where applicable).
- [ ] Traces defined for the main workflow, AI/agent calls, tool/MCP calls,
      external provider calls, and human approval flows.
- [ ] Audit trail makes every final output / state change traceable to input
      reference/hash, actor, decision path, intermediate outputs, validation
      results, AI/tool calls, human decisions, and final output hash/state.
- [ ] Audit records are append-only; immutability + retention defined in §14.
- [ ] The "what happened to this unit of work?" question is answerable from
      logs, traces, and the audit trail alone.
