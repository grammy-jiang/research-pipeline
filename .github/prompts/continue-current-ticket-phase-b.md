You are implementing the Daily AI Intelligence pipeline.

Before doing implementation work, perform startup audit.

Read:

- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/phase-a-to-b-transition-checklist.md
- docs/daily-ai-intelligence/phase-b-spec.md
- docs/daily-ai-intelligence/phase-b-implementation-backlog.md
- docs/daily-ai-intelligence/acceptance-gates.md
- docs/daily-ai-intelligence/startup-audit-policy.md
- docs/daily-ai-intelligence/copilot-execution-rules.md

## Startup audit

Audit every ticket marked `verified`, `complete`, or `audit_pass`.

If any Phase A ticket fails audit, reopen it and fix Phase A first.

Do not start Phase B unless Phase A is complete and B is unblocked.

## Select next ticket

After audit passes, identify the first pending or reopened Phase B ticket.

## Acceptance contract

Before implementation, create or update the ticket acceptance contract.

## DryRUN

List:

- files to add/modify
- tests/fixtures to create
- commands to run
- expected artifacts/telemetry
- positive, negative, and edge cases

## Implement

1. Write tests/fixtures.
2. Implement only the current ticket.
3. Run verification.
4. Fix failures.
5. Write proof pack.
6. Update `phase-status.yaml`.

## Phase B hard rules

- Memory is evidence, not truth.
- Re-check current evidence before ranking.
- Durable aliases and merges require review.
- Every memory write must include trigger, effect, rollback metadata, source IDs, and timestamp.
- Do not start Phase C.
- Do not implement Obsidian, dossiers, MCP expansion, or social sources.

## Final response format

```markdown
## Implementation State

Current phase:
Current ticket completed:
Verification run:
Status file updated:
Next pending/reopened ticket:
Do not start:
```
