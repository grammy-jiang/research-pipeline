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

## Output

`intermediate/clarifications.md` — the questions asked and the
answers/assumptions, ready to populate §3 and §4.9.

## Validation / failure policy

- Gate: high-impact decisions are resolved or assumed.
- Failure policy: `record_assumptions`.
