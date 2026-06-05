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
- 17.9 External Provider Boundary (+ Data Egress / External Model Use table)
- 17.10 Audit and Compliance Requirements
- 17.11 Security Failure Modes
- 17.12 Security Quality Gates (**verification table**, not checkboxes)

Include the required security events (auth_failed, permission_denied,
prompt_injection_suspected, tool_call_denied, trust_boundary_violation_detected,
external_provider_error, secret_access_denied, audit_record_integrity_failure).

### Data Egress / External Model Use table (§17.9, mandatory when external models are used)

Expand the §3 headline data-egress decision into a dedicated table (do **not**
merge it into the provider-abstraction choice). Each value ∈ {local_only,
external_allowed, external_allowed_with_redaction, hybrid_by_domain,
unknown_requires_user_review}; `unknown_requires_user_review` blocks
implementation planning.

```text
| Decision | Value | Source | Decision Evidence | Review Requirement | Reason |
| Can raw or projected source content leave the local trust boundary? | … |
| Which providers may receive content? | … |
| Is redaction required before model calls? | … |
| May logs contain raw source content? | No (default) | … |
| Can domain plugins override data-egress policy? | … |
```

**Raw source content is forbidden in logs by default.** For any system using
external model providers, the "May logs contain raw source content?" answer is
**No** unless the user explicitly opts in with justification. If the provider
SDK/wrapper may log prompt content (e.g. verbose logging), the architecture must
**require** redaction, a safe log level, a **log-snapshot test**, and a
**provider-wrapper redaction test** — never rely on operator configuration
alone. Add a §17.12 gate "raw source content never written to logs" (blocks
release) with those tests as the verification method. Logs must not become a
second, accidental data-egress channel.

### Security Quality Gates as a verification table (§17.12)

Render §17.12 as a **verification table**, never as unchecked `- [ ]`
checkboxes (a checkbox is ambiguous — requirement? done? TODO? — and misleads
the implementation-plan skill):

```text
| Security Gate | Required Implementation Evidence | Verification Method | Blocks Release? |
```

Every gate row states what must be true, the evidence that proves it, how it is
verified, and whether it blocks release. Keep each gate's wording **honest to
the chosen technology** — e.g. an append-only audit on an embedded file-based
store is "application-enforced (single-writer; no update/delete path exposed) +
hash-chain tamper-evident", never "no UPDATE/DELETE granted to the application
DB user" (that borrows a role/grant model the store does not have).

## Output

`intermediate/security_trust_boundaries.md` → populates §17.

## Validation / failure policy

- Gate: trust zones, the AI/LLM boundary, tool permissions, and security events
  are present; AI cannot mutate state without deterministic validation;
  external providers are isolated behind adapters; secrets strategy exists.
- Failure policy: `revise`.
