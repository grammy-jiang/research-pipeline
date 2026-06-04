# Rule Pack: Reliability Rules

Apply as a review gate (prompt 21).

- [ ] Timeouts are defined for every external and long-running call.
- [ ] A retry policy (with backoff/jitter) is defined where retries are safe.
- [ ] Fallback behavior is defined for AI, tool/MCP, and external-provider
      failures.
- [ ] Idempotency is considered for state-changing operations.
- [ ] Queue/job failure recovery is defined (re-run, resume, dead-letter).
- [ ] Partial failure handling is defined for multi-step workflows.

**Gate:** no critical workflow may lack failure handling; unchecked items block
§18.
