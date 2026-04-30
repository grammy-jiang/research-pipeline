You are performing the final completeness audit for the Daily AI Intelligence pipeline.

This is not new feature development.

Read first:

- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/final-audit-spec.md
- docs/daily-ai-intelligence/final-feature-inventory.md
- docs/daily-ai-intelligence/final-audit-checklist.md
- docs/daily-ai-intelligence/final-gap-triage-policy.md
- docs/daily-ai-intelligence/acceptance-gates.md
- docs/daily-ai-intelligence/startup-audit-policy.md
- docs/daily-ai-intelligence/copilot-execution-rules.md

## Step 1 — Status audit

Verify all A-G tickets are `audit_pass`.

If any ticket is not `audit_pass`, stop and reopen/fix that ticket.

## Step 2 — Feature inventory audit

For every item in `final-feature-inventory.md`, find evidence:

- implementation file(s)
- test file(s)
- CLI/MCP/skill surface where applicable
- acceptance contract
- proof pack
- validator/e2e coverage where applicable

## Step 3 — Traceability matrix

Create:

```text
docs/daily-ai-intelligence/final-traceability-matrix.md
```

Each row must include:

- feature/function
- phase/ticket
- implementation files
- tests
- user/agent-facing surface
- proof pack
- status: pass/fail/partial/not-applicable
- notes

## Step 4 — Gap register

Create:

```text
docs/daily-ai-intelligence/final-gap-register.md
```

For every failed/partial item include:

- gap ID
- feature
- original phase/ticket
- severity
- evidence
- required fix
- recommended ticket to reopen or create

## Step 5 — Run final gates

Run the final acceptance commands from:

```text
docs/daily-ai-intelligence/final-acceptance-gate.md
```

If commands fail, record the failure and stop.

## Step 6 — Final report

Create:

```text
docs/daily-ai-intelligence/final-completeness-audit-report.md
```

It must state whether the project is complete or not complete.

Do not mark complete if there are unresolved gaps.

## Hard rules

- Do not silently implement new features during audit.
- If a fix is required, create/reopen a specific ticket.
- Do not weaken tests to pass the audit.
- Do not ignore non-goal violations.
- Do not call the project complete unless the final gap register is empty and final gates pass.

End with:

## Final Audit State
All A-G tickets audit_pass:
Final gates run:
Gap register empty:
Traceability matrix written:
Final audit report written:
Project complete:
Next required action:
