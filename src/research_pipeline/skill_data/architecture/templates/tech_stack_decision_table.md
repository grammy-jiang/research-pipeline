# Tech Stack Decision Table Template

For each major technology choice, produce one row. Stacks are chosen for *this*
blueprint — never a universal default.

```markdown
| Decision | Recommendation | Alternatives | Rationale | Risks | Reversible? |
|---|---|---|---|---|---|
| Primary programming language | <choice> | <alts> | <fit> | <risk> | yes/no |
| Backend framework | <choice> | <alts> | <fit> | <risk> | yes/no |
| CLI framework (if needed) | <choice> | <alts> | <fit> | <risk> | yes/no |
| API style | <choice> | <alts> | <fit> | <risk> | yes/no |
| Job execution model | <choice> | <alts> | <fit> | <risk> | yes/no |
| Storage system | <choice> | <alts> | <fit> | <risk> | yes/no |
| Artifact storage | <choice> | <alts> | <fit> | <risk> | yes/no |
| Search / retrieval system | <choice> | <alts> | <fit> | <risk> | yes/no |
| Queue / background jobs | <choice> | <alts> | <fit> | <risk> | yes/no |
| Observability stack | <choice> | <alts> | <fit> | <risk> | yes/no |
| Testing framework | <choice> | <alts> | <fit> | <risk> | yes/no |
| Configuration / secrets | <choice> | <alts> | <fit> | <risk> | yes/no |
| LLM provider abstraction | <choice> | <alts> | <fit> | <risk> | yes/no |
| Agent orchestration approach | <choice> | <alts> | <fit> | <risk> | yes/no |
| MCP strategy | adopt/defer | <alts> | <fit> | <risk> | yes/no |
| Deployment target | <choice> | <alts> | <fit> | <risk> | yes/no |
```

## Decision criteria (state which drove each choice)

```text
fit to blueprint workflows; AI/LLM ecosystem maturity; team/user familiarity;
interface contract support; observability support; security fit; local
development simplicity; deployment simplicity; performance; cost; long-term
maintainability; reversibility
```

## Anti-default rule

> Never select a fixed stack (e.g. Python/FastAPI/PostgreSQL) as the default
> for all projects. Select it only when blueprint context, team constraints,
> ecosystem fit, and deployment requirements justify it. Mark the table
> **provisional** until the Traditional-vs-AI matrix and skill/MCP decision are
> done, then run the coherence review (prompt 12) and revise.
