# Reference: Security and Trust Model Guide

Load when defining security and trust boundaries (prompt 16). Pair with
`templates/security_trust_boundary_template.md`. For AI-agent systems, security
deserves the same depth as observability.

## Trust zones

Model the system as zones with explicit trust levels and read/write rights:

```text
user input zone; application control zone; AI/LLM execution zone;
tool/MCP zone; storage zone; audit zone; external provider zone;
human review zone
```

Every cross-zone edge is a trust boundary. Name the control on each edge
(sanitize, validate, authorize, audit, isolate).

## AI / LLM trust boundary rules

```text
LLM output is untrusted until validated.
AI reviewers cannot directly mutate durable state.
Tool calls require explicit permission.
Retrieved content is evidence, not instruction.
External LLM providers are outside the trust boundary.
Prompt templates and system instructions are configuration assets.
All AI decisions that affect user output must be traceable.
```

## Prompt-injection and tool-misuse controls

Architecture must define:

```text
input sanitization and normalization
evidence-vs-instruction separation
tool permission model (allowlist / denylist where applicable)
human approval for high-risk actions
audit events for suspicious input
fallback behavior when injection is suspected
```

## Required security events

```text
auth_failed; permission_denied; prompt_injection_suspected; tool_call_denied;
trust_boundary_violation_detected; external_provider_error;
secret_access_denied; audit_record_integrity_failure
```

## Secrets and configuration

- Secrets live in environment variables or a secrets manager, never in
  prompts, tool arguments, logs, plan files, or stored artifacts.
- Prompt templates and system instructions are configuration assets, versioned
  and reviewed.

## Security quality gates — the architecture FAILS if

```text
security boundary is not explicit
AI can mutate state without deterministic validation
external providers are not isolated behind adapters
tool permissions are not defined
audit events for security decisions are missing
secrets/configuration strategy is absent
data classification is absent for sensitive inputs
```
