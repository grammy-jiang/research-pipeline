# Feature Acceptance Contract: Add GitHub releases adapter with fixtures

## Feature ID

`A03_github_releases_adapter_with_fixtures`

## Goal

Convert registry-allowed GitHub release data into `IntelligenceEvent` records using offline fixtures.

Specifically ensure deterministic behavior for:

- fixture-backed offline polling
- release payload normalization into `IntelligenceEvent`
- malformed payload rejection
- rate-limit and empty-payload edge handling in fixture tests

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Validate stable event fields produced by the adapter from release JSON.
- Validate fixture loading behavior without network calls.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/sources/__init__.py`
- `src/research_pipeline/briefing/sources/base.py`
- `src/research_pipeline/briefing/sources/github_releases.py`
- `tests/unit/test_briefing_github_releases.py`
- `tests/fixtures/briefing/github/releases_normal.json`
- `tests/fixtures/briefing/github/releases_empty.json`
- `tests/fixtures/briefing/github/releases_malformed.json`
- `tests/fixtures/briefing/github/releases_rate_limited.json`

## Required Tests

- `tests/unit/test_briefing_github_releases.py`
	- normal fixture maps to expected event fields
	- empty fixture returns no events
	- malformed fixture is rejected deterministically
	- rate-limited fixture behavior is deterministic and explicit

## Required Fixtures

- `tests/fixtures/briefing/github/releases_normal.json`
- `tests/fixtures/briefing/github/releases_empty.json`
- `tests/fixtures/briefing/github/releases_malformed.json`
- `tests/fixtures/briefing/github/releases_rate_limited.json`

## Failure Cases To Cover

- missing repo metadata for non-fixture API path
- empty payload fixture
- malformed fixture payload shape
- deterministic handling of rate-limited payload fixture
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_github_releases.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- adapter outputs valid `IntelligenceEvent` values from fixture inputs;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A03_github_releases_adapter_with_fixtures.md
```
