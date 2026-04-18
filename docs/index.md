# research-pipeline

A production-grade, deterministic Python pipeline for searching, screening,
downloading, converting, and summarizing academic papers from arXiv, Google
Scholar, Semantic Scholar, OpenAlex, and DBLP.

## Overview

research-pipeline automates the full academic literature review workflow through
seven composable stages:

**plan → search → screen → download → convert → extract → summarize**

Each stage is idempotent and can be run independently, with full artifact
tracking via SHA-256 hashed manifests.

## Key capabilities

- **Multi-source search** across arXiv, Google Scholar, Semantic Scholar,
  OpenAlex, and DBLP with cross-source deduplication
- **Semantic re-ranking** via optional SPECTER2 embeddings
- **Quality evaluation** — composite scoring combining citation impact, venue
  reputation (CORE rankings), author h-index, and recency
- **Multi-backend PDF conversion** — 3 local + 5 cloud backends with
  multi-account rotation and cross-service fallback
- **MCP server** for AI agent integration — 21 tools, 15 resources, 6 prompts,
  auto-completions, and a harness-engineered research workflow with server-driven
  orchestration
- **Incremental runs** — SQLite global index deduplicates papers across runs

## Getting started

See the [User Guide](user-guide.md) for installation instructions,
configuration reference, and CLI usage examples.

## Architecture

For technical details on the stage-based pipeline design, data flow, and
cross-cutting concerns, see the [Architecture](architecture.md) documentation.

## Links

- [GitHub Repository](https://github.com/grammy-jiang/research-pipeline)
- [PyPI Package](https://pypi.org/project/research-pipeline/)
- [Changelog](changelog.md)
