# Feature Acceptance Contract: Add held-out agent evaluation tasks

## Feature ID
`G08_held_out_agent_evals`

## Goal
Add held-out agent evaluation tasks for Phase G harness hardening and agent workflows.

## In Scope
Workflow/MCP/skill/eval hardening only; tests first; offline tests; preserve A-F behavior; proof pack.

## Out of Scope
New sources, UI/dashboard, browser scraping, behavioral tracking, unrelated academic workflow changes, weakening existing governance.

## Expected Owned Files
- `tests/integration_offline/test_briefing_phase_g_agent_evals.py`
- `tests/fixtures/briefing/e2e/agent_eval/run_daily_brief/`
- `tests/fixtures/briefing/e2e/agent_eval/validate_malformed_report/`
- `tests/fixtures/briefing/e2e/agent_eval/record_feedback/`
- `tests/fixtures/briefing/e2e/agent_eval/export_obsidian/`
- `tests/fixtures/briefing/e2e/agent_eval/refuse_unsupported_source/`
- `tests/fixtures/briefing/e2e/agent_eval/paper_request_handoff/`

## Required Tests
- `tests/integration_offline/test_briefing_phase_g_agent_evals.py`

## Failure Cases
Invalid workflow state; non-namespaced MCP tool; MCP behavior diverges from CLI; unsafe write annotation; skill triggers on paper-only literature review; unsupported source expansion not refused; raw source dumps sent to cloud model; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/integration_offline/test_briefing_phase_g_agent_evals.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Acceptance Criteria
Tests pass; no scope creep; MCP namespace/governance preserved where applicable; skill/eval behavior verified where applicable; status and proof pack updated.
