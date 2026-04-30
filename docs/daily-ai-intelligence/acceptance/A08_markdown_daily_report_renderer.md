# Feature Acceptance Contract: Add Markdown daily report renderer

## Feature ID

`A08_markdown_daily_report_renderer`

## Goal

Render a template/extractive daily Markdown report from ranked clusters, including low-signal and no-news variants.

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

- `src/research_pipeline/briefing/report.py`
- `tests/unit/test_briefing_report.py`

## Required Tests

- `tests/unit/test_briefing_report.py`
  - populated daily brief contains required sections and feedback targets
  - low-signal variant emits short-day explanation
  - no-news variant emits fallback messaging
  - weekly synthesis link extraction respects `max_links`

## Required Fixtures

- None required.

## Failure Cases To Cover

- Empty clusters produce deterministic no-news fallback sections.
- Low-signal day is explicitly called out when material items are limited.
- Weekly synthesis truncates extracted links to configured maximum.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_report.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- outputs follow the Phase A template/extractive report behavior;
- `phase-status.yaml` records ticket status, proof pack, and audit fields;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/A08_markdown_daily_report_renderer.md
```
