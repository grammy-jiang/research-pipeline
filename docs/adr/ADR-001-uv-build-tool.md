# ADR-001: Uv as the Build and Dependency Manager

## Status
Accepted

## Date
2024 (inferred from project history)

## Context

Python projects traditionally use `pip` + `requirements.txt` or `pyproject.toml`
with `pip install -e .`. Managing development extras, lockfiles, and virtual
environments consistently across machines was error-prone.

`uv` (Astral) offers:
- A single tool for virtual environments, dependency resolution, lockfile
  generation, and script execution
- 10–100× faster than pip for cold installs
- Reproducible installs via `uv.lock`
- Drop-in `uv run <cmd>` wrapper for consistent execution

## Decision

Use `uv` exclusively as the project's build and dependency manager. All
development commands must be prefixed with `uv run`. The lockfile `uv.lock`
is committed to the repository.

## Consequences

**Positive:**
- Reproducible installs on all machines and in CI
- Fast bootstrapping
- Single command to set up dev environment: `uv sync --extra dev`

**Negative:**
- Requires `uv` to be installed before bootstrapping (one extra step for new
  contributors)
- Bare `python` or `pytest` commands will not use the project virtual environment

**Mitigation:** All docs, AGENTS.md, and CI scripts consistently use `uv run`.
