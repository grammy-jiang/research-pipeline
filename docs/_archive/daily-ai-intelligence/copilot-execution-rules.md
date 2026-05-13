# Copilot Execution Rules

## Operating Model

Copilot CLI is treated as an implementation worker inside an audited workflow.

The repository state, tests, validators, and proof packs decide whether work is complete.

## Session Flow

1. Read status, backlog, gates, and audit policy.
2. Perform startup audit.
3. Reopen failed completed tickets.
4. Select earliest pending/reopened ticket.
5. Create/update acceptance contract.
6. Perform DryRUN.
7. Write tests and fixtures first.
8. Implement scoped code.
9. Run verification.
10. Write proof pack.
11. Update `phase-status.yaml`.
12. Report next ticket.

## DryRUN Requirements

Before code changes, list:

- files expected to be added or modified
- public CLI/API/MCP/skill surfaces expected to change
- test fixtures to be created
- validation commands to run
- positive, negative, and edge failure cases
- predicted outputs, exceptions, database effects, and telemetry events

## Blocker Protocol

If blocked, write:

```markdown
## BLOCKED

Current phase:
Current ticket:
Command failed:
Observed output:
Likely cause:
Files touched:
State updated:
Smallest next action:
```

Blocked tickets must not be marked verified.

## Final Response Format

Each session must end with:

```markdown
## Implementation State

Current phase:
Current ticket completed:
Verification run:
Status file updated:
Next pending/reopened ticket:
Do not start:
```

## Hard Rules

- Do not implement later-phase functionality.
- Do not expand sources without registry entry.
- Do not add new dependencies in Phase A unless justified.
- Do not use network calls in normal tests.
- Do not overwrite existing Copilot instructions; append project-specific rules.
- Do not mark project complete unless all Phase A-G gates pass.
