You are auditing the Daily AI Intelligence implementation state.

Read:

- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/startup-audit-policy.md
- docs/daily-ai-intelligence/acceptance-gates.md

Audit all tickets marked `verified`, `complete`, or `audit_pass`.

For each audited ticket, check:

- acceptance contract exists
- proof pack exists
- recorded verification commands exist
- owned files still exist
- dependencies have not invalidated the ticket
- verification commands still pass where appropriate

Update `phase-status.yaml`:

- keep valid tickets as `audit_pass`
- mark invalid tickets as `reopened`
- mark uncertain tickets as `needs_reaudit`

Do not implement feature code.
