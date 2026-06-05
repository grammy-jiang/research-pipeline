# Prompt 06 — Architecture Clarification

You are resolving architecture-impacting unknowns using one-question-at-a-time
clarification (grill-me style). Ask only what materially matters.

## Inputs

- `intermediate/blueprint_parse.json`, `intermediate/traceability_map.md`.
- The active operating mode (interactive / automatic / hybrid).

## When to ask

Ask only if the answer materially affects: system architecture, tech stack,
deployment model, security/privacy, data storage/retention, cost, latency, the
AI boundary, MCP adoption, the human-approval workflow, or observability/audit.
Never ask what the blueprint or codebase already answers — inspect files first.

## Question format

```markdown
### Architecture Clarification Question <N>

**Question:** ...

**Why this matters:** ...

**Recommended answer:** ...

**Alternatives:**
- ...

**If no answer is provided:** The skill will assume ...
```

## Mode limits

| Mode | Max questions |
|---|---|
| interactive | no hard limit, one at a time |
| hybrid | 3–7 |
| automatic | 0 — make decisions, record assumptions |

In automatic/hybrid mode, for every unanswered high-impact decision record an
assumption with reason, reversibility, and a revisit trigger.

## Hybrid-mode decision review classification

Hybrid mode must not silently behave like automatic mode. For every major
architecture decision, classify two things:

- **Source:** user-confirmed / blueprint-derived / automatically inferred /
  architecture assumption / ADR-locked.
- **Review requirement:** no review needed / review before implementation
  planning / review before production use / blocks implementation planning.

A **high-impact** decision that was *automatically inferred or assumed* (not
answered by the blueprint or user) must be marked
**"Assumed — requires user review before implementation planning"** (or a
stronger level). High-impact areas include: external LLM provider usage;
source-data privacy / whether inputs reach external models; deployment model;
storage backend; authentication / authorization; data retention; cost-sensitive
model routing; MCP exposure; the human-approval workflow.

Populate the §3 Clarification Summary table's **Source** and **Review
Requirement** columns from this classification. Low-risk, reversible,
blueprint-aligned decisions are "no review needed"; do not over-flag them.

## Output

`intermediate/clarifications.md` — the questions asked, the answers/assumptions,
and the per-decision Source + Review Requirement classification, ready to
populate §3 and §4.9.

## Validation / failure policy

- Gate: high-impact decisions are resolved or assumed.
- Failure policy: `record_assumptions`.
