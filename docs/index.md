# research-pipeline

A production-grade, deterministic Python pipeline for searching, screening,
downloading, converting, and summarizing academic papers — plus a daily AI
intelligence briefing system.

## Two pipelines in one package

### Academic research pipeline
Automates the full literature review workflow through seven composable,
idempotent stages:

```
plan → search → screen → download → convert → extract → summarize
```

### Daily AI intelligence briefing
Polls configured sources, deduplicates, ranks and generates a concise daily
intelligence brief:

```
poll_sources → rank_events → generate_daily → validate_report
```

Both pipelines share the same LLM layer, rate limiting, retry, and config system.

## Key capabilities

- **Multi-source search** across arXiv, Google Scholar, Semantic Scholar,
  OpenAlex, and DBLP with cross-source deduplication
- **Semantic re-ranking** via optional SPECTER2 embeddings
- **Quality evaluation** — composite scoring combining citation impact, venue
  reputation (CORE rankings), author h-index, and recency
- **Multi-backend PDF conversion** — 3 local + 5 cloud backends with
  multi-account rotation and cross-service fallback
- **MCP server** for AI agent integration — 64 tools, 21 resources, 6 prompts,
  auto-completions, and a harness-engineered research workflow
- **Multi-tier memory** — working memory, episodic memory (SQLite), knowledge
  graph, and case-based reasoning for self-improving research
- **4-layer confidence architecture** — L1 fast signal → L2 adaptive →
  L3 DINCO calibration → L4 selective verification
- **Incremental runs** — SQLite global index deduplicates papers across runs
- **Adaptive stopping** — query-adaptive retrieval stopping criteria

## Getting started

See the [User Guide](user-guide.md) for installation, configuration, and CLI usage.

For a task-oriented introduction to common workflows, see the
[User Manual](user-manual.md).

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](user-guide.md) | Installation, configuration reference, all CLI commands |
| [User Manual](user-manual.md) | Task-based guide: common workflows step by step |
| [System Design](system-design.md) | Architecture, data flow, both pipelines, all sub-systems |
| [Developer Guide](developer-guide.md) | Contributor onboarding, adding stages/backends/sources |
| [API Reference](api-reference.md) | All 66 CLI commands + 64 MCP tools + 21 resources |
| [Data Model](data-model.md) | Pydantic domain models, SQLite schemas, stage output formats |
| [Security Model](security-model.md) | HC1–HC6 constraints, MCP guard, taint tracking |
| [Testing Strategy](testing-strategy.md) | Test pyramid, VCR cassettes, CI gates |
| [Operations Runbook](operations-runbook.md) | Deploy, monitor, troubleshoot, maintain |
| [Implementation Plan](implementation-plan.md) | Current state, open work, out-of-scope items |
| [ADRs](adr/README.md) | Architecture Decision Records |
| [Changelog](changelog.md) | Version history |
| [Latest audit](audit-deep-research-compliance-2026-05-17.md) | Compliance vs. comprehensive deep-research design (2026-05-17) |

## Links

- [GitHub Repository](https://github.com/grammy-jiang/research-pipeline)
- [PyPI Package](https://pypi.org/project/research-pipeline/)
