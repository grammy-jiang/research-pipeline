# Feature Acceptance Contract: Add replay diagnosis docs and final all-phase gate

## Feature ID
`G09_replay_diagnosis_and_final_gate`

## Goal
Add replay diagnosis docs and final all-phase gate for Phase G harness hardening and agent workflows.

## In Scope
Workflow/MCP/skill/eval hardening only; tests first; offline tests; preserve A-F behavior; proof pack.

## Out of Scope
New sources, UI/dashboard, browser scraping, behavioral tracking, unrelated academic workflow changes, weakening existing governance.

## Expected Owned Files
- `docs/daily-ai-intelligence/replay-diagnosis.md`
- `docs/daily-ai-intelligence/final-acceptance-gate.md`
- `tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py`

## Required Tests
- `tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py`

## Failure Cases
Invalid workflow state; non-namespaced MCP tool; MCP behavior diverges from CLI; unsafe write annotation; skill triggers on paper-only literature review; unsupported source expansion not refused; raw source dumps sent to cloud model; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/integration_offline/test_briefing_phase_g_replay_diagnosis.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Acceptance Criteria
Tests pass; no scope creep; MCP namespace/governance preserved where applicable; skill/eval behavior verified where applicable; status and proof pack updated.
