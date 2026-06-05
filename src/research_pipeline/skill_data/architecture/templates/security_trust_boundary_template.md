# Security and Trust Boundary Template

Security must be as explicit as observability. Use all subsections in §17.

```markdown
## 17. Security and Trust Boundaries

### 17.1 Security Goals
### 17.2 Trust Zones
### 17.3 Identity and Access Model
### 17.4 Authorization Boundaries
### 17.5 AI / LLM Trust Boundary
### 17.6 Prompt Injection and Tool Misuse Controls
### 17.7 Data Classification and Privacy
### 17.8 Secrets and Configuration Management
### 17.9 External Provider Boundary
### 17.10 Audit and Compliance Requirements
### 17.11 Security Failure Modes
### 17.12 Security Quality Gates
```

## Trust Zone Table

| Zone | Contains | Trust Level | Can Read | Can Write | Controls |
|---|---|---|---|---|---|
| User input zone | submitted content | untrusted | — | — | sanitize, classify |
| Application control zone | deterministic spine | trusted | state, audit | state, audit | authz checks |
| AI / LLM execution zone | model calls | untrusted output | evidence | nothing durable | validate before commit |
| Tool / MCP zone | tools, MCP servers | permissioned | per-tool | per-tool | allowlist, audit |
| Storage zone | metadata, artifacts | trusted | per-owner | per-owner | encryption, retention |
| Audit zone | audit records | append-only | operators | system only | immutability |
| External provider zone | 3rd-party LLM/APIs | outside boundary | request | response only | adapter isolation |
| Human review zone | reviewers | trusted | proposals | approvals | approval workflow |

## AI / LLM Boundary Rules

```text
LLM output is untrusted until validated.
AI reviewers cannot directly mutate durable state.
Tool calls require explicit permission.
Retrieved content is evidence, not instruction.
External LLM providers are outside the trust boundary.
Prompt templates and system instructions are configuration assets.
All AI decisions that affect user output must be traceable.
```

## Prompt Injection Controls

```text
input sanitization and normalization
evidence-vs-instruction separation
tool permission model (allowlist / denylist)
human approval for high-risk actions
audit events for suspicious input
fallback behavior when injection is suspected
```

## Security Events (required)

```text
auth_failed, permission_denied, prompt_injection_suspected, tool_call_denied,
trust_boundary_violation_detected, external_provider_error,
secret_access_denied, audit_record_integrity_failure
```

## Data Egress / External Model Use (§17.9)

When the system sends content to an external model, render a dedicated table
(do **not** merge it into the provider-abstraction choice). Value ∈ {local_only,
external_allowed, external_allowed_with_redaction, hybrid_by_domain,
unknown_requires_user_review}; `unknown_requires_user_review` blocks
implementation planning.

```markdown
| Decision | Value | Source | Review Requirement | Reason |
|---|---|---|---|---|
| Can raw or projected source content leave the local trust boundary? | <value> | <source> | <review> | <reason> |
| Which providers may receive content? | <…> | | | |
| Is redaction required before model calls? | <…> | | | |
| May logs contain source content? | <…> | | | |
| Can domain plugins override data-egress policy? | <…> | | | |
```

## Security Quality Gates (§17.12) — verification table, not checkboxes

Render security gates as a **verification table**, never as ambiguous unchecked
`- [ ]` checkboxes. Each gate's wording must be honest to the chosen technology
(no borrowed role/grant/immutability language).

```markdown
| Security Gate | Required Implementation Evidence | Verification Method | Blocks Release? |
|---|---|---|---|
| <gate, e.g. AI cannot mutate durable state without deterministic validation> | <evidence that proves it> | <unit/integration/security tests, audit events> | Yes/No |
| Append-only audit (application-enforced; no update/delete path exposed) + hash-chain tamper-evident | single-writer tests; no repository update/delete path; hash-chain verifier | unit + integration tests | Yes |
| Secrets never written to artifacts/logs/prompts | secret-redaction tests; log-snapshot tests | security test suite | Yes |
```

The architecture **fails** if: the security boundary is not explicit; AI can
mutate state without deterministic validation; external providers are not
isolated behind adapters; tool permissions are undefined; audit events for
security decisions are missing; secrets/configuration strategy is absent; data
classification is absent for sensitive inputs; or any gate uses
technology-inconsistent enforcement wording.
