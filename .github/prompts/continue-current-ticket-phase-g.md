Implement Phase G.

Read:
- docs/daily-ai-intelligence/phase-status.yaml
- docs/daily-ai-intelligence/phase-f-to-g-transition-checklist.md
- docs/daily-ai-intelligence/phase-g-spec.md
- docs/daily-ai-intelligence/phase-g-implementation-backlog.md
- docs/daily-ai-intelligence/acceptance-gates.md
- docs/daily-ai-intelligence/startup-audit-policy.md
- docs/daily-ai-intelligence/copilot-execution-rules.md

Audit all verified/complete/audit_pass tickets. If A/B/C/D/E/F fails audit, fix the earlier phase first.

Select first pending/reopened G ticket. Create/update acceptance contract before implementation. DryRUN. Tests/fixtures first. Implement only current ticket. Verify. Write proof pack. Update phase-status.yaml.

Hard rules: brief_* MCP namespace only; MCP mirrors stable CLI; no new sources; no UI; no weakening of governance; held-out agent evals required; do not call all phases complete until final gate passes.

End with:
## Implementation State
Current phase:
Current ticket completed:
Verification run:
Status file updated:
Next pending/reopened ticket:
Do not start:
