# Checklist: Security

- [ ] Security goals stated.
- [ ] Trust zones explicit (table), every cross-zone edge names a control.
- [ ] Identity/access and authorization boundaries defined.
- [ ] AI/LLM trust boundary rules applied (LLM output untrusted until
      validated; AI cannot mutate durable state; retrieved content is evidence,
      not instruction; external providers outside the boundary).
- [ ] Prompt-injection and tool-misuse controls defined.
- [ ] Data classification and privacy defined for sensitive inputs.
- [ ] Secrets/configuration strategy defined (env/secrets-manager only; never
      in prompts, logs, tool args, or artifacts).
- [ ] External providers isolated behind adapters.
- [ ] Required security events present (auth_failed, permission_denied,
      prompt_injection_suspected, tool_call_denied,
      trust_boundary_violation_detected, external_provider_error,
      secret_access_denied, audit_record_integrity_failure).
- [ ] Security failure modes and security quality gates stated.
- [ ] **Data egress decision** present when external models are used: §3
      records whether source/projected content may leave the local trust
      boundary (external_allowed / …_with_redaction / local_only /
      hybrid_by_domain / unknown_requires_user_review), separate from the
      provider-abstraction choice, and §17.9 reflects it.
- [ ] No residual technology-inconsistent security claims (whole-document scan,
      not just the tech-stack section).
- [ ] §17.12 security quality gates are a **verification table** (Security Gate ·
      Required Implementation Evidence · Verification Method · Blocks Release?),
      not ambiguous unchecked `- [ ]` checkboxes; each gate's wording is honest
      to the chosen technology.
- [ ] §17.9 includes the Data Egress / External Model Use table when external
      models are used.

**Hard fails:** AI mutates state without deterministic validation; secrets in
artifacts; external providers not isolated; tool permissions undefined; external
models used with no data-egress decision; security gates rendered as ambiguous
unchecked checkboxes.
