# Feature Acceptance Contract: Add reversible preference update engine

## Feature ID
`D05_preference_update_engine`

## Goal
Add reversible preference update engine for Phase D explicit feedback.

## In Scope
Explicit feedback only; tests first; offline tests; reversible/auditable preferences where applicable; proof pack.

## Out of Scope
Behavioral feedback, click/read-time tracking, Phase E+, dossiers, social sources, MCP, UI.

## Expected Owned Files
- `src/research_pipeline/briefing/preference_update.py`
- `tests/unit/test_briefing_preference_update.py`

## Required Tests
- `tests/unit/test_briefing_preference_update.py`

## Failure Cases
Malformed target ID; unsupported signal; insufficient feedback; conflicting feedback; stale target; rollback; no behavioral influence.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_preference_update.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; explicit feedback only; reversible changes where applicable; status and proof pack updated.
