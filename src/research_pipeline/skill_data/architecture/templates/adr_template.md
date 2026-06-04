# ADR Template

One file per decision under `adr/ADR-NNNN-<slug>.md`. Index them in §21 of the
architecture document.

```markdown
# ADR-NNNN: <Decision Title>

## Status

Proposed / Accepted / Rejected / Superseded

## Context

<The forces at play: blueprint drivers, constraints, NFRs, and the problem this
decision resolves. Link the blueprint section(s) and any clarification.>

## Decision

<The decision, stated in one or two sentences.>

## Alternatives Considered

- <Alternative A> — <why not>
- <Alternative B> — <why not>

## Consequences

<Positive and negative consequences, including downstream impact on other
sections (interfaces, data, security, observability, deployment).>

## Risk

<Key risk this decision introduces or accepts.>

## Reversal Cost

Low / Medium / High

## Confidence

Low / Medium / High

## Review Date

<When to revisit, or the trigger that forces a revisit.>

## Supersedes

ADR-NNNN  <!-- Required only when this ADR replaces a prior decision; omit for
new decisions. When superseding, set the old ADR's Status to "Superseded" and
link back here. -->
```

Generate ADRs for high-impact decisions: runtime architecture, tech stack,
storage strategy, AI boundary, MCP adoption or deferral, observability/audit
model, deployment model, and security/trust-boundary model.
