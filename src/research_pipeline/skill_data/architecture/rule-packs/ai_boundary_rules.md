# Rule Pack: AI Boundary Rules

Apply as a review gate (prompt 21).

- [ ] AI decisions are bounded (scope of what the model may decide is explicit).
- [ ] Tool permissions are explicit (allowlist; state-changing tools gated).
- [ ] Human approval gates are defined for high-risk actions.
- [ ] LLM failures have fallback paths.
- [ ] AI outputs are validated deterministically before any state change.
- [ ] Retrieved content is treated as evidence, not instruction.
- [ ] AI decisions affecting user output are traceable (correlation IDs +
      audit).

**Gate:** if AI can mutate durable state without a deterministic validation
gate, the architecture FAILS (block §6/§11/§17).
