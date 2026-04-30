# Feature Acceptance Contract: Add deterministic ranking and tie-breakers

## Feature ID

`A07_deterministic_ranking_and_tie_breakers`

## Goal

Rank clusters using Phase A deterministic scoring and stable tie-breakers.

Specifically ensure deterministic behavior for:

- rank score composition
- minimum-score filtering
- low-information filtering
- tie-break order consistency

## In Scope

- Implement only this ticket's Phase A behavior.
- Use existing project conventions: uv, Typer, Pydantic, pathlib, logging, ruff, mypy.
- Add or update tests before implementation.
- Keep normal tests offline.
- Write or update a proof pack after verification passes.
- Validate ranking tie-breakers follow the Phase A ordering contract.

## Out of Scope

- Later-phase functionality.
- Browser scraping.
- Social-source ingestion.
- Obsidian export.
- MCP tool implementation before CLI stabilization.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/rank.py`
- `tests/unit/test_briefing_rank.py`

## Required Tests

- `tests/unit/test_briefing_rank.py`
	- higher score clusters rank first
	- low-information clusters are filtered
	- tie-breakers are deterministic when scores match
	- `max_items` and `min_rank_score` options are enforced

## Required Fixtures

- None required.

## Failure Cases To Cover

- Empty cluster list returns empty ranking.
- Low-information clusters are excluded.
- Tie-break ordering is stable across runs.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_rank.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- ranking behavior and tie-break order are covered by deterministic tests;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A07_deterministic_ranking_and_tie_breakers.md
```
