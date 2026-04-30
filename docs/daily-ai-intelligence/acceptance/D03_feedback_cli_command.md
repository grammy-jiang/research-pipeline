# Feature Acceptance Contract: Add CLI command brief feedback

## Feature ID
`D03_feedback_cli_command`

## Goal
Add CLI command brief feedback for Phase D explicit feedback.

## In Scope
Explicit feedback only; tests first; offline tests; reversible/auditable preferences where applicable; proof pack.

## Out of Scope
Behavioral feedback, click/read-time tracking, Phase E+, dossiers, social sources, MCP, UI.

## Expected Owned Files
- `src/research_pipeline/cli/cmd_brief.py`
- `tests/unit/test_briefing_cli_feedback.py`

## Required Tests
- `tests/unit/test_briefing_cli_feedback.py`

## Failure Cases
Malformed target ID; unsupported signal; insufficient feedback; conflicting feedback; stale target; rollback; no behavioral influence.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_cli_feedback.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; explicit feedback only; reversible changes where applicable; status and proof pack updated.
