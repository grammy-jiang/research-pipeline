You are auditing whether Phase A is complete enough to start Phase B.

Read:

- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/acceptance-gates.md
- docs/daily-ai-intelligence/startup-audit-policy.md

Check:

1. Every Phase A ticket A01-A12 is `audit_pass`.
2. Every A-ticket has an acceptance contract.
3. Every A-ticket has a proof pack.
4. Every A-ticket has recorded verification commands.
5. Phase A gate commands are recorded or runnable.
6. `phases.A.status` is complete.
7. `phases.B.status` is in_progress.

If any condition fails, do not start Phase B.
