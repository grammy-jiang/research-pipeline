# Feature Acceptance Contract: Add CLI command group and fixed artifact layout

## Feature ID

`A10_cli_command_group_and_artifact_layout`

## Goal

Add `research-pipeline brief ...` commands and the fixed Phase A workspace artifact layout.

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

- `src/research_pipeline/briefing/artifacts.py`
- `src/research_pipeline/cli/cmd_brief.py`
- `src/research_pipeline/cli/app.py`
- `tests/unit/test_briefing_artifacts.py`
- `tests/unit/test_briefing_cli.py`

## Required Tests

- `tests/unit/test_briefing_artifacts.py`
- `tests/unit/test_briefing_cli.py`

## Required Fixtures

- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover

- Missing required inputs.
- Empty input where applicable.
- Malformed input where applicable.
- Policy-denied or unsupported configuration where applicable.
- Deterministic output ordering where applicable.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_artifacts.py tests/unit/test_briefing_cli.py -xvs
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
docs/daily-ai-intelligence/proofs/A10_cli_command_group_and_artifact_layout.md
```
