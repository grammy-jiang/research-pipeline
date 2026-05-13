# Feature Acceptance Contract: Add weekly feedback section and e2e tests

## Feature ID
`D08_feedback_weekly_section_and_e2e`

## Goal
Add weekly feedback section and e2e tests for Phase D explicit feedback.

## In Scope
Explicit feedback only; tests first; offline tests; reversible/auditable preferences where applicable; proof pack.

## Out of Scope
Behavioral feedback, click/read-time tracking, Phase E+, dossiers, social sources, MCP, UI.

## Expected Owned Files
- `src/research_pipeline/briefing/weekly.py`
- `tests/unit/test_briefing_weekly_feedback.py`
- `tests/integration_offline/test_briefing_phase_d_feedback_e2e.py`
- `tests/fixtures/briefing/e2e/feedback_loop/`

## Required Tests
- `tests/unit/test_briefing_weekly_feedback.py`
- `tests/integration_offline/test_briefing_phase_d_feedback_e2e.py`

## Failure Cases
Malformed target ID; unsupported signal; insufficient feedback; conflicting feedback; stale target; rollback; no behavioral influence.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_weekly_feedback.py tests/integration_offline/test_briefing_phase_d_feedback_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; explicit feedback only; reversible changes where applicable; status and proof pack updated.
