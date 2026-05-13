# Feature Acceptance Contract: Add feedback event store

## Feature ID
`D02_feedback_store`

## Goal
Add feedback event store for Phase D explicit feedback.

## In Scope
Explicit feedback only; tests first; offline tests; reversible/auditable preferences where applicable; proof pack.

## Out of Scope
Behavioral feedback, click/read-time tracking, Phase E+, dossiers, social sources, MCP, UI.

## Expected Owned Files
- `src/research_pipeline/briefing/feedback_store.py`
- `tests/unit/test_briefing_feedback_store.py`
- `tests/fixtures/briefing/feedback/feedback_empty.sqlite`

## Required Tests
- `tests/unit/test_briefing_feedback_store.py`

## Failure Cases
Malformed target ID; unsupported signal; insufficient feedback; conflicting feedback; stale target; rollback; no behavioral influence.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_feedback_store.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; explicit feedback only; reversible changes where applicable; status and proof pack updated.
