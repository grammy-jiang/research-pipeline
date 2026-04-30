# Phase B Migration Guide

## Keep Phase A files

Do not delete:

- `docs/daily-ai-intelligence/acceptance/A*.md`
- `docs/daily-ai-intelligence/proofs/A*.md`
- Phase A tests and fixtures
- Phase A status records
- Phase A prompts

## Update `copilot-instructions.md`

Replace the old Daily AI Intelligence Phase A block with:

```text
.github/copilot-instructions.daily-ai-intelligence-phase-b.append-or-replace.md
```

or append it at the end as the current override.

Keep the repository's base instructions unchanged.

## Update status

After audit confirms Phase A is complete:

- set `current_phase: B`
- set `current_ticket: B01_topic_memory_models`
- set `phases.A.status: complete`
- set `phases.B.status: in_progress`
- merge B01-B08 from `phase-b-status-addendum.yaml`

## Start Copilot

```bash
copilot --autopilot --allow-all --max-autopilot-continues 20 \
  -p "$(cat prompts/daily-ai-intelligence/continue-current-ticket-phase-b.md)"
```
