You are implementing the Daily AI Intelligence pipeline.

Before doing any implementation work, perform the startup completeness audit.

Read these files first:

- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/implementation-backlog.md
- docs/daily-ai-intelligence/acceptance-gates.md
- docs/daily-ai-intelligence/startup-audit-policy.md
- docs/daily-ai-intelligence/copilot-execution-rules.md
- docs/daily-ai-intelligence/phase-a-spec.md

## Step 1 — Startup audit

For every ticket marked `verified`, `complete`, or `audit_pass`:

1. Check that its acceptance contract exists.
2. Check that its proof pack exists.
3. Check that its verification commands are recorded.
4. Check that owned files still exist.
5. Re-run relevant verification commands when needed.
6. Check whether dependencies changed after `last_verified_commit`.

If any completed ticket fails audit:

- mark it as `reopened`;
- record the reason;
- fix the earliest reopened ticket first;
- do not start new work.

## Step 2 — Select next ticket

After audit passes, identify the first pending or reopened ticket in the current unblocked phase.

## Step 3 — Feature acceptance contract

Before implementation, create or update the ticket acceptance contract.

The contract must define:

- feature goal
- in-scope behavior
- out-of-scope behavior
- required tests
- fixtures
- validators
- failure cases
- verification commands
- acceptance criteria

Do not implement until this contract exists.

## Step 4 — DryRUN

Produce a DryRUN:

- files expected to be added or modified
- public CLI/API/MCP/skill surfaces expected to change
- test fixtures to be created
- validation commands to run
- positive, negative, and edge failure cases
- predicted outputs, exceptions, database effects, and telemetry events

## Step 5 — Implement

Follow this sequence:

1. Write or update tests and fixtures.
2. Implement only the current ticket.
3. Run verification commands.
4. Fix failures.
5. Generate a proof pack.
6. Update `phase-status.yaml`.

## Hard rules

- Do not trust `done` without audit.
- Do not mark a ticket verified without a proof pack.
- Do not implement a feature before defining how it will be tested.
- Do not start later-phase work.
- Do not add new dependencies in Phase A unless explicitly justified.
- Do not make network calls in normal tests.
- Do not call the project complete unless all Phase A-G gates pass.

## Final response format

End with:

```markdown
## Implementation State

Current phase:
Current ticket completed:
Verification run:
Status file updated:
Next pending/reopened ticket:
Do not start:
```
