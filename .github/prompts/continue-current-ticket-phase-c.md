You are implementing Phase C of the Daily AI Intelligence pipeline.

Read `phase-status.yaml`, `phase-b-to-c-transition-checklist.md`, `phase-c-spec.md`, `phase-c-implementation-backlog.md`, `acceptance-gates.md`, `startup-audit-policy.md`, and `copilot-execution-rules.md`.

Audit all `verified`, `complete`, or `audit_pass` tickets. If any A/B ticket fails, reopen it and fix the earlier phase first.

Select the first pending/reopened Phase C ticket. Create/update acceptance contract before implementation. Produce DryRUN. Write tests/fixtures first. Implement only current ticket. Run verification. Write proof pack. Update status.

Hard rules: write only under configured vault path; preserve wiki-links; do not overwrite human notes without matching generated frontmatter ID; dry-run required; do not start Phase D.

End with:

## Implementation State
Current phase:
Current ticket completed:
Verification run:
Status file updated:
Next pending/reopened ticket:
Do not start:
