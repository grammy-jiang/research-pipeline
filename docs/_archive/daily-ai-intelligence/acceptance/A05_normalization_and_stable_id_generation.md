# Feature Acceptance Contract: Add normalization and stable ID generation

## Feature ID

`A05_normalization_and_stable_id_generation`

## Goal

Implement deterministic normalization and stable IDs for events, content hashes, dedup keys, and cluster IDs.

Specifically ensure deterministic behavior for:

- title normalization
- URL canonicalization
- stable event ID generation
- stable content hash generation
- dedup key precedence policy
- cluster ID generation

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Verify stable hashing returns deterministic identifiers.
- Verify dedup-key selection follows Phase A precedence rules.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/normalize.py`
- `tests/unit/test_briefing_normalize.py`

## Required Tests

- `tests/unit/test_briefing_normalize.py`
	- canonical URL normalization removes `utm_*` params and sorts query keys
	- stable hash and event IDs are deterministic
	- dedup key precedence matches Phase A policy
	- cluster ID generation is deterministic with expected prefix

## Required Fixtures

- None required.

## Failure Cases To Cover

- URL normalization edge cases (mixed case scheme/host, tracking params).
- Missing stronger IDs falls back to deterministic weaker key.
- Different input content yields different stable hashes.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_normalize.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- normalization and stable ID helpers are covered by deterministic tests;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A05_normalization_and_stable_id_generation.md
```
