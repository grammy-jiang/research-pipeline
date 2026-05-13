# Phase G Spec — Harness Hardening and Agent Workflows

## Preconditions

A01-A12, B01-B08, C01-C08, D01-D08, E01-E08, and F01-F09 are `audit_pass`; phases A/B/C/D/E/F are complete; G is in_progress.

## Purpose

Make the daily briefing pipeline replayable, auditable, and agent-safe.

## Scope

- briefing workflow state model
- stage verifiers
- telemetry/replay/diagnosis documentation
- source/network allowlist integration with workflow state
- report validation gates and budget enforcement surfaced to agents
- namespaced `brief_*` MCP schemas/tools/resources
- dedicated bundled `daily-ai-intelligence` skill
- tool-scope governance and annotations
- held-out agent evaluation tasks
- final all-phase acceptance gate

## MCP tools

`brief_poll_sources`, `brief_rank_events`, `brief_generate_daily`, `brief_validate_report`, `brief_export_obsidian`, `brief_record_feedback`, `brief_generate_dossier`, `brief_weekly_synthesis`.

MCP tools mirror stable CLI behavior.

## MCP resources

Read-only resources for daily brief, ranked clusters, telemetry, validation results, and workflow state.

## Skill

Add `src/research_pipeline/skill_data/daily-ai-intelligence/` with `SKILL.md`, `config.toml`, and references.

## Held-out evals

Run daily brief, validate malformed report, record feedback, export Obsidian notes, refuse unsupported source expansion, and hand off paper-only request to academic skill.

## Non-goals

No new source expansion, UI/dashboard, behavioral tracking, browser scraping, new cloud summarization behavior, or weakening of A-F governance.

## Completion

G01-G09 audit_pass; A-F tests still pass; MCP tools/resources work and are namespaced; skill triggers/non-triggers are correct; held-out agent evals pass; final gate passes.
