# Feature Acceptance Contract: Add offline end-to-end tests for normal, low-signal, and no-news days

## Feature ID

`A12_offline_e2e_tests_normal_low_signal_no_news`

## Goal

Prove Phase A works offline across normal, low-signal, no-news, duplicate, and validation-failure scenarios.

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `tests/integration_offline/test_briefing_phase_a_e2e.py`
- `tests/fixtures/briefing/e2e/normal/`
- `tests/fixtures/briefing/e2e/low_signal/`
- `tests/fixtures/briefing/e2e/no_news/`

## Required Tests

- `tests/integration_offline/test_briefing_phase_a_e2e.py`

## Required Fixtures

- `tests/fixtures/briefing/e2e/normal/`
- `tests/fixtures/briefing/e2e/low_signal/`
- `tests/fixtures/briefing/e2e/no_news/`

## Failure Cases To Cover

- Missing required inputs.
- Empty input where applicable.
- Malformed input where applicable.
- Policy-denied or unsupported configuration where applicable.
- Deterministic output ordering where applicable.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- outputs follow the Phase A spec where applicable;
- `phase-status.yaml` records the proof pack and verification commands;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A12_offline_e2e_tests_normal_low_signal_no_news.md
```
