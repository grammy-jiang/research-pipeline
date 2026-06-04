# Reference: ADR Guidance

Load when generating or updating ADRs (prompt 20). Use `templates/adr_template.md`
for the record shape.

## When to write an ADR

Write one ADR per high-impact decision:

```text
runtime architecture; tech stack; storage strategy; AI boundary;
MCP adoption or deferral; observability and audit model; deployment model;
security/trust-boundary model
```

A decision is "high-impact" when it is costly to reverse, constrains many other
sections, or carries security/privacy/cost/latency consequences.

## Status lifecycle

```text
Proposed → Accepted → (later) Superseded
Proposed → Rejected   (decision considered and declined)
```

## Supersession rules (update runs)

```text
Do not silently overwrite an ADR.
When a decision changes, create a NEW ADR and set the old one's Status to
  "Superseded", linking forward to the replacement.
The new ADR's "Supersedes" field names the old ADR id. Omit "Supersedes" only
  for brand-new decisions.
Create a new ADR (rather than editing) for any high-impact change.
```

## Numbering

- Sequential, zero-padded: `ADR-0001`, `ADR-0002`, …
- File name: `adr/ADR-NNNN-<kebab-slug>.md`.
- Index every ADR in §21 of the architecture document with its status.

## Quality bar

- Context names the blueprint driver(s) and constraint(s).
- At least two real alternatives, each with a reason for rejection.
- Consequences list downstream impact on other sections.
- Reversal Cost and Confidence are stated; a low-confidence, high-reversal-cost
  decision should carry an explicit review trigger.
