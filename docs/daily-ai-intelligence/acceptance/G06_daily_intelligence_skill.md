# Feature Acceptance Contract: Add bundled daily-ai-intelligence skill

## Feature ID
`G06_daily_intelligence_skill`

## Goal
Add bundled daily-ai-intelligence skill for Phase G harness hardening and agent workflows.

## In Scope
Workflow/MCP/skill/eval hardening only; tests first; offline tests; preserve A-F behavior; proof pack.

## Out of Scope
New sources, UI/dashboard, browser scraping, behavioral tracking, unrelated academic workflow changes, weakening existing governance.

## Expected Owned Files
- `src/research_pipeline/skill_data/daily-ai-intelligence/SKILL.md`
- `src/research_pipeline/skill_data/daily-ai-intelligence/config.toml`
- `src/research_pipeline/skill_data/daily-ai-intelligence/references/command-reference.md`
- `src/research_pipeline/skill_data/daily-ai-intelligence/references/source-policy.md`
- `src/research_pipeline/skill_data/daily-ai-intelligence/references/report-templates.md`
- `src/research_pipeline/skill_data/daily-ai-intelligence/references/feedback-loop.md`
- `src/research_pipeline/skill_data/daily-ai-intelligence/references/troubleshooting.md`
- `tests/unit/test_skill_daily_ai_intelligence.py`

## Required Tests
- `tests/unit/test_skill_daily_ai_intelligence.py`

## Failure Cases
Invalid workflow state; non-namespaced MCP tool; MCP behavior diverges from CLI; unsafe write annotation; skill triggers on paper-only literature review; unsupported source expansion not refused; raw source dumps sent to cloud model; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/unit/test_skill_daily_ai_intelligence.py -xvs
uv run ruff check src/research_pipeline/briefing src/research_pipeline/mcp_server tests/
uv run mypy src/
```

## Acceptance Criteria
Tests pass; no scope creep; MCP namespace/governance preserved where applicable; skill/eval behavior verified where applicable; status and proof pack updated.
