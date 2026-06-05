# Prompt 16 — Security and Trust Boundaries

You are defining security with the same depth as observability.

## Inputs

- `intermediate/c4_views.md`, `intermediate/data_contracts.md`,
  `intermediate/ai_responsibility_matrix.md`.
- `templates/security_trust_boundary_template.md`,
  `references/security_trust_model_guide.md`.

## Instructions

Produce all of §17:

- 17.1 Security Goals
- 17.2 Trust Zones (table: Zone · Contains · Trust Level · Can Read · Can Write
  · Controls)
- 17.3 Identity and Access Model
- 17.4 Authorization Boundaries
- 17.5 AI / LLM Trust Boundary (apply the boundary rules)
- 17.6 Prompt Injection and Tool Misuse Controls
- 17.7 Data Classification and Privacy
- 17.8 Secrets and Configuration Management
- 17.9 External Provider Boundary
- 17.10 Audit and Compliance Requirements
- 17.11 Security Failure Modes
- 17.12 Security Quality Gates

Include the required security events (auth_failed, permission_denied,
prompt_injection_suspected, tool_call_denied, trust_boundary_violation_detected,
external_provider_error, secret_access_denied, audit_record_integrity_failure).

In §17.9 (External Provider Boundary), reflect the **data-egress decision** from
§3 (external_allowed / external_allowed_with_redaction / local_only /
hybrid_by_domain / unknown_requires_user_review): state exactly what content may
cross the boundary, what is redacted, and any local-only fallback. Keep claims
honest — do not credit a chosen technology with enforcement it does not provide;
say "application-enforced" / "tamper-evident" rather than borrowing another
technology's permission model.

## Output

`intermediate/security_trust_boundaries.md` → populates §17.

## Validation / failure policy

- Gate: trust zones, the AI/LLM boundary, tool permissions, and security events
  are present; AI cannot mutate state without deterministic validation;
  external providers are isolated behind adapters; secrets strategy exists.
- Failure policy: `revise`.
