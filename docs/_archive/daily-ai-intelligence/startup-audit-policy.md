# Startup Audit Policy

Every Copilot CLI session must begin with a startup completeness audit.

The agent must not start new implementation work until the audit completes.

## Audit Scope

For every ticket marked `verified`, `complete`, or `audit_pass` in `phase-status.yaml`, check:

1. The ticket has an acceptance contract.
2. The ticket has a proof pack.
3. The ticket has recorded verification commands.
4. The files claimed by the ticket still exist.
5. The relevant tests still pass when needed.
6. The relevant validators still pass when needed.
7. The ticket was verified against the current commit or is marked for re-audit.
8. Dependencies have not changed after the ticket was last verified.

## Audit Result

Each completed ticket must be assigned one of:

- `audit_pass`
- `audit_failed`
- `needs_reaudit`
- `missing_proof`
- `missing_acceptance_contract`

## Reopen Rule

If a completed ticket fails audit, change its status to `reopened`.

Do not continue to later tickets until reopened earlier tickets are fixed.

## No Blind Trust Rule

The agent must not say "this is already done" only because the status file says so.

Status is advisory. Verification is authoritative.
