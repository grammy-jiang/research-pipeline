# Rule Pack: Security Rules

Apply as a review gate (prompt 21).

- [ ] Trust zones are explicit and every cross-zone edge names a control.
- [ ] Authorization boundaries are explicit.
- [ ] Secrets handling is defined (env/secrets-manager only; never in prompts,
      logs, tool args, or artifacts).
- [ ] External providers are isolated behind adapters.
- [ ] Prompt-injection controls exist for every LLM-facing input.
- [ ] Security-relevant decisions emit audit events.
- [ ] Data classification exists for sensitive inputs.

**Gate:** unchecked items block §17 and the security-test row in §19;
secrets-in-artifact findings are always FAIL.
