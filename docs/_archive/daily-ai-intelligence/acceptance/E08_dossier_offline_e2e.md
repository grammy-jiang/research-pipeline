# Feature Acceptance Contract: Add offline dossier e2e tests

## Feature ID
`E08_dossier_offline_e2e`

## Goal
Add offline dossier e2e tests for Phase E hot-topic dossiers.

## In Scope
Manual dossier workflow only; primary artifact gate; one-topic focus; evidence timeline; factuality labels; validation; tests first; offline tests; proof pack.

## Out of Scope
Automatic scheduling, Phase F source expansion, social sources, MCP expansion, UI, general literature review, raw source dump summarization.

## Expected Owned Files
- `tests/integration_offline/test_briefing_phase_e_dossier_e2e.py`
- `tests/fixtures/briefing/e2e/dossier_manual/`
- `tests/fixtures/briefing/e2e/dossier_no_primary_artifact/`
- `tests/fixtures/briefing/e2e/dossier_long_rejection/`

## Required Tests
- `tests/integration_offline/test_briefing_phase_e_dossier_e2e.py`

## Failure Cases
Missing cluster; missing primary artifact; missing evidence URL; unsupported factuality label; overlong dossier; multi-topic expansion; no network in normal tests.

## Verification Commands
```bash
uv run pytest tests/integration_offline/test_briefing_phase_e_dossier_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; manual dossier only; primary artifact required; claims labeled; validator passes/fails correctly; status and proof pack updated.
