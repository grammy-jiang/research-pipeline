# Feature Acceptance Contract: Add source registry loader and validation

## Feature ID

`A02_source_registry_loader_and_validation`

## Goal

Load and validate Phase A source registry configuration.

Specifically ensure deterministic behavior for:

- loading JSON and TOML registry formats
- validating unique `source_id` values
- applying enabled-source budget limits
- enforcing Phase A source-boundary guardrails

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Validate file-not-found and malformed registry handling.
- Validate source boundary checks for non-Phase-A access methods.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/registry.py`
- `tests/unit/test_briefing_registry.py`
- `docs/daily-ai-intelligence/source-registry.phase-a.example.yaml`

## Required Tests

- `tests/unit/test_briefing_registry.py`
	- missing registry path raises `FileNotFoundError`
	- duplicate source IDs are rejected
	- JSON and TOML registry loading works
	- enabled source selection respects `max_sources_per_run`
	- boundary guard rejects unreviewed non-Phase-A sources

## Required Fixtures

- None required.

## Failure Cases To Cover

- Missing registry path raises a deterministic file error.
- Duplicate `source_id` entries are rejected.
- Invalid source configuration is rejected by model validation.
- Unreviewed non-Phase-A access methods are rejected by boundary guard.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_registry.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- source registry loader behavior is verified by unit tests;
- boundary checks for Phase A scope are verified;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A02_source_registry_loader_and_validation.md
```
