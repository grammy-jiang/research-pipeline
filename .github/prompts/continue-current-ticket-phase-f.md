Implement Phase F.

Read:
- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/phase-e-to-f-transition-checklist.md
- docs/daily-ai-intelligence/phase-f-spec.md
- docs/daily-ai-intelligence/phase-f-implementation-backlog.md
- docs/daily-ai-intelligence/acceptance-gates.md
- docs/daily-ai-intelligence/startup-audit-policy.md
- docs/daily-ai-intelligence/copilot-execution-rules.md

Audit all verified/complete/audit_pass tickets. If A/B/C/D/E fails audit, fix the earlier phase first.

Select first pending/reopened F ticket. Create/update acceptance contract before implementation. DryRUN. Tests/fixtures first. Implement only current ticket. Verify. Write proof pack. Update phase-status.yaml.

Hard rules: no scraping; no firehose; source disabled by default; registry entry required; fixtures before enablement; compare with/without source; do not start Phase G.

End with:
## Implementation State
Current phase:
Current ticket completed:
Verification run:
Status file updated:
Next pending/reopened ticket:
Do not start:
