# Feature Acceptance Contract: Add exact deduplication

## Feature ID

`A06_exact_deduplication`

## Goal

Deduplicate normalized events by exact deterministic keys.

Specifically ensure deterministic behavior for:

- exact dedup by `dedup_key`
- fallback merge via canonical URL and normalized title
- stable primary-event selection within a cluster
- deterministic output ordering

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Validate duplicate penalty behavior for merged clusters.
- Validate cluster IDs are deterministic from dedup keys.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/dedup.py`
- `tests/unit/test_briefing_dedup.py`

## Required Tests

- `tests/unit/test_briefing_dedup.py`
	- exact-key duplicates merge into one cluster
	- URL/title fallback merging is deterministic
	- primary event selection follows quality preference
	- output cluster ordering is deterministic

## Required Fixtures

- None required.

## Failure Cases To Cover

- Empty input returns no clusters.
- Competing duplicate records resolve to a stable primary event.
- Duplicate penalty increases with merged event count.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_dedup.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- deduplication and clustering behavior is covered by deterministic tests;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A06_exact_deduplication.md
```
