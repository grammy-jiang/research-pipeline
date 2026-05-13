# Feature Acceptance Contract: Add RSS/Atom adapter with fixtures

## Feature ID

`A04_rss_atom_adapter_with_fixtures`

## Goal

Convert conservative RSS/Atom fields into `IntelligenceEvent` records using offline fixtures.

Specifically ensure deterministic behavior for:

- fixture-backed RSS parsing
- fixture-backed Atom parsing
- malformed XML rejection
- empty feed handling

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Validate normalized event fields produced from RSS and Atom entries.
- Validate malformed XML handling is deterministic.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/sources/rss_atom.py`
- `tests/unit/test_briefing_rss_atom.py`
- `tests/fixtures/briefing/rss/rss_normal.xml`
- `tests/fixtures/briefing/rss/rss_empty.xml`
- `tests/fixtures/briefing/rss/rss_malformed.xml`
- `tests/fixtures/briefing/atom/atom_normal.xml`

## Required Tests

- `tests/unit/test_briefing_rss_atom.py`
	- RSS normal fixture maps to expected event fields
	- Atom normal fixture maps to expected event fields
	- empty RSS fixture returns no events
	- malformed RSS fixture raises parser error

## Required Fixtures

- `tests/fixtures/briefing/rss/rss_normal.xml`
- `tests/fixtures/briefing/rss/rss_empty.xml`
- `tests/fixtures/briefing/rss/rss_malformed.xml`
- `tests/fixtures/briefing/atom/atom_normal.xml`

## Failure Cases To Cover

- Empty feed fixture yields no events.
- Malformed XML fixture raises deterministic parser error.
- Unsupported/empty entry payloads are safely ignored.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_rss_atom.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- adapter outputs valid `IntelligenceEvent` values from RSS and Atom fixtures;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A04_rss_atom_adapter_with_fixtures.md
```
