# User Story Template

Use this skeleton for every major user story in §10. A story is **testable** —
its acceptance criteria are observable and its E2E seeds are derivable.

```markdown
## User Story: <Story Name>

**As a:** <role>
**I want:** <goal>
**So that:** <value>

### Preconditions
- <what must be true before the story starts; tie to architecture state>

### Main Flow
1. <user action>
2. <system response>
3. <observable result>

### Alternative Flows
- <variation of the happy path the architecture supports>

### Failure / Recovery Flows
- <what the user does and sees when a step fails; tie to the architecture
  failure/recovery model and §15>

### User-Visible States
- <user-visible state → architecture lifecycle state / condition flag / audit
  event (must resolve to the architecture state model)>

### Acceptance Criteria
- <observable + testable outcome: "Given … When … Then …" or a checkable
  assertion of an outcome the user can see>

### E2E Scenario Seeds
- <one-line pointer to the §20 Gherkin seed(s) this story produces>
```

## Rules

- Every major story includes **all** of: preconditions, main flow, alternative
  flows, failure/recovery flows, user-visible states, acceptance criteria, E2E
  scenario seeds.
- **User-Visible States must resolve to the architecture state model.** If a
  story needs a state the architecture lacks, do not invent it — record it as
  architecture feedback (§21).
- **Acceptance Criteria are specific and testable** (observable outcome +
  condition), never vague ("works well", "is fast").
- Stay at the experience level — **no** exact CLI flags, API schemas, visual
  layout, or copy.
- Cover reviewer and agent/MCP stories when those actors exist in the
  architecture.
