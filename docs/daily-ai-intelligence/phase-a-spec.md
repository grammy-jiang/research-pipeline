# Daily AI Intelligence — Phase A Spec

## Purpose

Phase A builds the smallest reliable daily Markdown briefing pipeline.

It proves the thin daily report is useful before memory, Obsidian, dossiers, social sources, MCP tools, or richer agent workflows are added.

## Phase A Scope

Phase A includes:

- `research_pipeline/briefing/` package skeleton
- `IntelligenceEvent`
- `BriefingSourceConfig`
- source registry
- GitHub releases adapter
- RSS/Atom adapter
- HTTP politeness metadata
- fixed artifact layout
- stable ID generation
- raw JSONL per source
- normalized event JSONL
- exact deduplication
- deterministic ranking and tie-breakers
- template/extractive Markdown report
- validation
- telemetry JSONL
- CLI commands:
  - `brief poll`
  - `brief rank`
  - `brief generate-daily`
  - `brief validate`
  - `brief run`
- offline tests for normal, low-signal, and no-news days

## Phase A Non-Goals

Phase A does not include:

- Obsidian graph
- hot-topic dossiers
- feedback learning
- behavioral tracking
- full knowledge graph
- X/Twitter
- Reddit
- Bluesky
- Hacker News
- full MCP tools
- cloud summarization of raw source dumps
- hidden LLM judgment in ranking or default report generation

## Fixed Artifact Layout

```text
workspace/briefings/YYYY-MM-DD/
  source_registry_snapshot.json
  raw/
    <source_id>.jsonl
  normalized/
    events.jsonl
  clusters/
    clusters.jsonl
  ranked/
    ranked_clusters.jsonl
  reports/
    daily.md
  validation/
    validation.json
  telemetry.jsonl
```

## Command Contracts

| Command | Input | Output |
|---|---|---|
| `brief poll` | source registry | `source_registry_snapshot.json`, `raw/*.jsonl`, `normalized/events.jsonl`, `telemetry.jsonl` |
| `brief rank` | `normalized/events.jsonl` | `clusters/clusters.jsonl`, `ranked/ranked_clusters.jsonl` |
| `brief generate-daily` | `ranked/ranked_clusters.jsonl` | `reports/daily.md` |
| `brief validate` | `reports/daily.md`, `ranked/ranked_clusters.jsonl` | `validation/validation.json` |
| `brief run` | source registry | all Phase A artifacts, in command-contract order |

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Validation failed |
| 2 | Source polling failed but partial output exists |
| 3 | Configuration or source registry error |
| 4 | Unexpected internal error |

## Stable ID Policy

```text
event_id =
  stable hash of source_id + source_native_id when source_native_id exists,
  otherwise stable hash of source_id + canonical_url

content_hash =
  stable hash of normalized title + canonical_url + published_at + summary_hint

dedup_key =
  strongest available key in order:
    source_native_id
    canonical_url
    repo/tag for GitHub release events
    RSS/Atom guid/id
    normalized title

cluster_id =
  stable hash of primary dedup_key
```

Normalize titles, URLs, and timestamps before hashing.

## Ranking Formula

```text
rank_score =
  source_class_weight
+ trust_weight
- noise_weight
+ primary_artifact_bonus
+ watchlist_match_bonus
- hype_penalty
- duplicate_penalty
```

## Ranking Tie-Breakers

```text
rank_score desc
source_class_weight desc
published_at desc
trust_weight desc
title asc
event_id asc
```

## Phase A LLM Policy

Phase A report generation is template-based and extractive by default.

LLM summarization is disabled unless explicitly enabled by configuration.

If enabled:

- send only ranked evidence packs
- never send raw unclustered source dumps
- preserve source URLs and evidence labels
- run deterministic validation after report generation

## Low-Signal / No-News Policy

The system must not pad the daily brief.

If fewer than 6 high-quality items exist, generate a shorter low-signal brief.

A no-news day is valid output, not a pipeline failure.

## Completion Gate

Phase A is complete only when:

- A01-A12 are `audit_pass`
- unit tests pass
- offline integration/e2e tests pass
- lint/type checks pass
- artifact layout is stable
- normal, low-signal, and no-news days are covered
