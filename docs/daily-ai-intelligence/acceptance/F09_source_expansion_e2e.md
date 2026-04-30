# Feature Acceptance Contract: Add source-expansion offline e2e and report-comparison tests

## Feature ID
`F09_source_expansion_e2e`

## Goal
Add source-expansion offline e2e and report-comparison tests for Phase F source expansion.

## In Scope
One source/governance unit only; registry policy; disabled-by-default behavior; offline fixtures; parser/evaluation tests; no network in normal tests; proof pack.

## Out of Scope
Browser scraping, social firehose, automatic source enablement, Phase G MCP/skill hardening, UI, behavioral tracking.

## Expected Owned Files
- `tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py`
- `tests/fixtures/briefing/e2e/source_expansion_baseline/`
- `tests/fixtures/briefing/e2e/source_expansion_with_new_source/`

## Required Tests
- `tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py`

## Failure Cases
Missing registry entry; enabled-by-default noisy source; missing retention policy; missing cadence/rate-limit policy; malformed fixture; unsupported access method; report bloat; no side-by-side comparison where applicable.

## Verification Commands
```bash
uv run pytest tests/integration_offline/test_briefing_phase_f_source_expansion_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria
Tests pass; no scraping; source disabled by default where applicable; registry/retention/cadence/rate-limit policy defined; status and proof pack updated.
