# Feature Acceptance Contract: Link manual dossier output to daily brief and optional archive path

## Feature ID
`E07_dossier_archive_linking`

## Goal
Link manual dossier output to daily brief and optional archive path for Phase E hot-topic dossiers.

## In Scope
Manual dossier workflow only; primary artifact gate; one-topic focus; evidence timeline; factuality labels; validation; tests first; offline tests; proof pack.

## Out of Scope
Automatic scheduling, Phase F source expansion, social sources, MCP expansion, UI, general literature review, raw source dump summarization.

## Expected Owned Files
- `src/research_pipeline/briefing/report.py`
- `src/research_pipeline/briefing/dossier.py`
- `tests/unit/test_briefing_dossier_linking.py`

## Required Tests
- `tests/unit/test_briefing_dossier_linking.py`

## Failure Cases
Missing cluster; missing primary artifact; missing evidence URL; unsupported factuality label; overlong dossier; multi-topic expansion; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/unit/test_briefing_dossier_linking.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; manual dossier only; primary artifact required; claims labeled; validator passes/fails correctly; status and proof pack updated.
