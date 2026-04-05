# Architecture

## Overview

research-pipeline is a deterministic, stage-based pipeline for academic paper
research. It automates the workflow of finding, evaluating, downloading,
converting, and summarizing scholarly papers.

## Pipeline stages

The pipeline processes a research topic through 7 sequential stages:

```
plan → search → screen → download → convert → extract → summarize
```

### 1. Plan

Normalizes a natural language research topic into a structured `QueryPlan`
containing must-have terms, nice-to-have terms, negative terms, and query
variants optimized for different source APIs.

- Input: research topic string
- Output: `plan/query_plan.json`
- Implementation: `src/research_pipeline/cli/cmd_plan.py`

### 2. Search

Executes the query plan against one or more sources (arXiv, Google Scholar) in
parallel, collecting candidate papers with metadata (title, abstract, authors,
categories, dates).

- Input: `query_plan.json`
- Output: `search/candidates.jsonl`, `search/raw/*.xml`
- Implementation: `src/research_pipeline/cli/cmd_search.py`
- Sources: `src/research_pipeline/sources/`

### 3. Screen

Two-stage relevance filtering. First pass uses BM25 heuristic scoring against
the query terms. Optional second pass uses an LLM judge for deeper evaluation.

- Input: `search/candidates.jsonl`
- Output: `screen/cheap_scores.jsonl`, `screen/shortlist.json`
- Implementation: `src/research_pipeline/cli/cmd_screen.py`
- Scoring: `src/research_pipeline/screening/heuristic.py`

### 4. Download

Downloads PDFs for shortlisted papers with rate limiting, retries, and atomic
writes. Respects arXiv's polite-mode guidelines.

- Input: `screen/shortlist.json`
- Output: `download/pdf/*.pdf`, `download/download_manifest.jsonl`
- Implementation: `src/research_pipeline/cli/cmd_download.py`

### 5. Convert

Converts downloaded PDFs to Markdown using Docling, preserving document
structure (headings, tables, equations).

- Input: `download/pdf/*.pdf`
- Output: `convert/markdown/*.md`, `convert/convert_manifest.jsonl`
- Implementation: `src/research_pipeline/cli/cmd_convert.py`
- Backend: `src/research_pipeline/conversion/docling_backend.py`

### 6. Extract

Chunks Markdown documents by heading structure and token limits. Builds a BM25
index for retrieval during summarization.

- Input: `convert/markdown/*.md`
- Output: `extract/*.extract.json`
- Implementation: `src/research_pipeline/cli/cmd_extract.py`

### 7. Summarize

Generates per-paper summaries with evidence citations and a cross-paper
synthesis report identifying agreements, disagreements, and open questions.

- Input: `extract/*.extract.json`
- Output: `summarize/*.summary.json`, `summarize/synthesis.json`,
  `summarize/synthesis.md`
- Implementation: `src/research_pipeline/cli/cmd_summarize.py`

## Cross-cutting concerns

### Configuration

Configuration is loaded from multiple sources with this precedence:

1. Environment variables (highest priority)
2. TOML config file (`config.toml`)
3. Built-in defaults (lowest priority)

Configuration schemas are defined in `src/research_pipeline/config/models.py`
and defaults in `src/research_pipeline/config/defaults.py`.

### Manifest tracking

Every pipeline run produces a `run_manifest.json` that records:

- Run ID, timestamps, and configuration snapshot
- Per-stage records (start/end times, status, errors)
- Artifact records with SHA-256 hashes for reproducibility

Implementation: `src/research_pipeline/storage/manifests.py`

### Rate limiting

arXiv enforces rate limiting on API access. The pipeline uses a global,
thread-safe rate limiter with a 3-second hard floor and configurable delay
(default 5 seconds).

Implementation: `src/research_pipeline/arxiv/rate_limit.py`

### Caching

HTTP responses are cached locally with a configurable TTL (default 24 hours)
to avoid redundant API calls during development and re-runs.

Implementation: `src/research_pipeline/infra/cache.py`

### Logging

All logging uses Python's `logging` module with structured JSONL output to both
console and file (`logs/pipeline.jsonl`).

Implementation: `src/research_pipeline/infra/logging.py`

## Multi-source architecture

The pipeline supports multiple paper sources through the `SearchSource` protocol:

```
SearchSource (protocol)
├── ArxivSource      — arXiv API (Atom XML)
├── ScholarlySource  — Google Scholar (free, via scholarly library)
└── SerpAPISource    — Google Scholar (paid, via SerpAPI)
```

Sources run in parallel with cross-source deduplication by arXiv ID and
normalized title.

## MCP server

The MCP server (`mcp_server/`) wraps pipeline functionality into 10 tools
accessible via the Model Context Protocol. Tools are thin adapters that delegate
to the same logic used by the CLI. The server uses FastMCP with stdio transport.

## Directory structure

```
src/research_pipeline/
├── cli/            # Typer CLI commands (one file per stage)
├── models/         # Pydantic domain models
├── config/         # Configuration loading and schemas
├── arxiv/          # arXiv API client, XML parser, dedup, rate limiter
├── sources/        # Multi-source search adapters
├── screening/      # BM25 heuristic scoring, LLM judge interface
├── download/       # Rate-limited PDF downloader
├── conversion/     # PDF→Markdown backends
├── extraction/     # Markdown chunking and retrieval
├── summarization/  # Per-paper and cross-paper synthesis
├── pipeline/       # Orchestrator and stage sequencing
├── storage/        # Workspace management, manifests, artifacts
├── infra/          # Cache, HTTP, logging, hashing, clock, paths
└── llm/            # LLM provider interface (experimental)
```
