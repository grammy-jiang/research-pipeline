# Rule Pack: Observability Rules

Apply as a review gate (prompt 21).

- [ ] Each workflow has correlation IDs that propagate end to end.
- [ ] Critical decisions emit audit events.
- [ ] Agent/tool calls are traceable.
- [ ] Quality scores (if any) are reproducible from logged inputs.
- [ ] An operator can answer "what happened to this unit of work?" from logs,
      traces, and the audit trail alone.
- [ ] Metrics cover throughput, latency, errors, and cost at minimum.

**Gate:** unchecked items block §16 and the observability-test row in §19.
