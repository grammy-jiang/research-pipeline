# Phase C Migration Guide

Keep Phase A/B files. Replace old Daily AI Intelligence Phase B Copilot block with `.github/copilot-instructions.daily-ai-intelligence-phase-c.append-or-replace.md` or append it as current override.

Merge `phase-c-status-addendum.yaml` into `phase-status.yaml`; set current_ticket to C01. Append the Phase C gate addendum to `acceptance-gates.md`.

Start:

```bash
copilot --autopilot --allow-all --max-autopilot-continues 20 \
  -p "$(cat prompts/daily-ai-intelligence/continue-current-ticket-phase-c.md)"
```
