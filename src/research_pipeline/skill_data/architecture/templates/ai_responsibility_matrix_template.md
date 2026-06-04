# Traditional Software vs AI Responsibility Matrix Template

This matrix is mandatory (§6). One row per responsibility. Mark the owner(s)
with ✓ and leave others as `–`.

```markdown
| Responsibility | Traditional Software | AI / LLM | Skill | MCP Server | Human | Notes |
|---|---:|---:|---:|---:|---:|---|
| Workflow orchestration | ✓ | – | – | – | – | deterministic sequencing |
| Durable state writes | ✓ | – | – | – | – | only after validation |
| Audit recording | ✓ | – | – | – | – | append-only |
| Authorization / permissions | ✓ | – | – | – | – | deterministic |
| Input validation | ✓ | – | – | – | – | schema + sanitization |
| Language-/judgment-heavy step | – | ✓ | – | – | – | output validated downstream |
| Reusable external tool/data access | – | – | – | ✓ | – | only if MCP justified |
| Reasoning workflow prompt | – | – | ✓ | – | – | invoked by human/agent |
| High-risk approval | – | – | – | – | ✓ | HITL gate |
```

## Core rule

> Deterministic control, state, storage, audit, security, workflow
> transitions, and interface contracts belong to traditional software unless
> explicitly justified. AI owns only judgment-heavy, language-heavy, or
> reasoning-heavy tasks.

## Quality gate

> AI components must not mutate durable state directly unless deterministic
> validation and audit controls exist. If a row places state mutation under
> AI, it must show a downstream deterministic validation gate before commit.
