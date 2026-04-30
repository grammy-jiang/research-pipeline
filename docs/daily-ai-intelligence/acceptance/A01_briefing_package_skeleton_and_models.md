# Feature Acceptance Contract: Add briefing package skeleton and models

## Feature ID

`A01_briefing_package_skeleton_and_models`

## Goal

Create `src/research_pipeline/briefing/` and core Phase A domain models.

Specifically ensure Phase A has a stable model contract for:

- source registry entries (`BriefingSourceConfig`)
- normalized events (`IntelligenceEvent`)
- ranked clusters (`BriefingCluster`)
- run metadata (`BriefingRunMetadata`)

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation (test-first).
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Verify that the package exports the core Phase A model surface.
- Verify model validation behavior for required fields and access-method constraints.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/__init__.py`
- `src/research_pipeline/briefing/models.py`
- `tests/unit/test_briefing_models.py`

## Required Tests

- `tests/unit/test_briefing_models.py`
	- package exports include core briefing models
	- `IntelligenceEvent` rejects empty title
	- `BriefingSourceConfig` validates `github_releases` requirements
	- `BriefingSourceConfig` validates `rss_atom` requirements
	- `BriefingSourceConfig` validates `manual` requirements

## Required Fixtures

- None required.

## Failure Cases To Cover

- Empty event title is rejected.
- `github_releases` source without repo/api/fixture is rejected.
- `rss_atom` source without feed/fixture is rejected.
- `manual` source without manual items is rejected.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_models.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- model contracts required by Phase A exist and are importable from `research_pipeline.briefing`;
- validation rules listed above are enforced by tests;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A01_briefing_package_skeleton_and_models.md
```
