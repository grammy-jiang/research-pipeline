# ADR-003: Stage-Based Idempotent Pipeline Architecture

## Status
Accepted

## Date
2024

## Context

Academic literature search is a long-running, multi-step process where each
step can fail independently (network timeouts, API rate limits, PDF download
failures, conversion errors). Users need to be able to:
- Resume a failed run without re-doing completed work
- Re-run a single stage with different parameters
- Inspect intermediate results at any stage

Two architectural styles were considered:
1. **Monolithic pipeline** — one command, all or nothing
2. **Stage-based pipeline** — each stage is independent and idempotent

## Decision

The research paper pipeline is decomposed into **7 independent, idempotent stages**:
`plan → search → screen → download → convert → extract → summarize`.

Each stage:
- Reads from a well-defined input directory under `<workspace>/<run_id>/<stage>/`
- Writes to a well-defined output directory
- Is fully **idempotent** — re-running a stage produces the same output given
  the same input (skips already-completed work)
- Records its status in `run_manifest.json` with SHA-256 hashes for
  integrity verification

The `research-pipeline run` command simply orchestrates all 7 stages in sequence.

## Consequences

**Positive:**
- Any stage can be re-run independently after a failure
- Intermediate results can be inspected at any point
- New stages can be added without restructuring the existing pipeline
- CI tests can test each stage in isolation

**Negative:**
- More disk space used (all intermediate artefacts are kept)
- Users must understand `--run-id` to resume interrupted runs
- Adding a stage requires updating the orchestrator, CLI, MCP server, and tests

**Note:** The daily AI Intelligence briefing pipeline follows the same pattern
with 4 stages: `poll_sources → rank_events → generate_daily → validate_report`.
