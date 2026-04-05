# Architecture

## Overview

research-pipeline is a deterministic, stage-based pipeline for academic paper
research. It automates the workflow of finding, evaluating, downloading,
converting, and summarizing scholarly papers.

| Property | Value |
|----------|-------|
| Language | Python 3.12+ |
| Build system | uv (`uv_build` backend) |
| CLI framework | Typer |
| Data models | Pydantic v2 |
| Package import | `research_pipeline` |
| CLI command | `research-pipeline` |

## Pipeline stages

The pipeline processes a research topic through 7 sequential stages. Each stage
is **idempotent** — it can be re-run without side effects.

```
plan → search → screen → download → convert → extract → summarize
```

### 1. Plan

Normalizes a natural language research topic into a structured `QueryPlan`
containing must-have terms, nice-to-have terms, negative terms, and query
variants optimized for different source APIs.

| Property | Value |
|----------|-------|
| Input | Research topic string |
| Output | `plan/query_plan.json` (`QueryPlan` model) |
| Implementation | `src/research_pipeline/cli/cmd_plan.py` |
| Model | `src/research_pipeline/models/query_plan.py` |

### 2. Search

Executes the query plan against one or more sources (arXiv, Google Scholar) in
parallel, collecting candidate papers with metadata (title, abstract, authors,
categories, dates).

| Property | Value |
|----------|-------|
| Input | `plan/query_plan.json` |
| Output | `search/candidates.jsonl` (`CandidateRecord` per line), `search/raw/*.xml` |
| Implementation | `src/research_pipeline/cli/cmd_search.py` |
| Sources | `src/research_pipeline/sources/` (`ArxivSource`, `ScholarlySource`, `SerpAPISource`) |
| Model | `src/research_pipeline/models/candidate.py` |

### 3. Screen

Two-stage relevance filtering. First pass uses BM25 heuristic scoring against
the query terms. Optional second pass uses an LLM judge for deeper evaluation.

| Property | Value |
|----------|-------|
| Input | `search/candidates.jsonl` |
| Output | `screen/cheap_scores.jsonl`, `screen/shortlist.json` |
| Implementation | `src/research_pipeline/cli/cmd_screen.py` |
| Scoring | `src/research_pipeline/screening/heuristic.py` |
| Models | `src/research_pipeline/models/screening.py` (`CheapScoreBreakdown`, `RelevanceDecision`) |

### 4. Download

Downloads PDFs for shortlisted papers with rate limiting, retries, and atomic
writes. Respects arXiv's polite-mode guidelines.

| Property | Value |
|----------|-------|
| Input | `screen/shortlist.json` |
| Output | `download/pdf/*.pdf`, `download/download_manifest.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_download.py` |
| Model | `src/research_pipeline/models/download.py` (`DownloadManifestEntry`) |

### 5. Convert

Converts downloaded PDFs to Markdown using a pluggable backend system.
Three backends are available out of the box:

| Backend | Package | License | Strengths |
|---------|---------|---------|----------|
| `docling` | `docling>=2.0` | MIT | Great table/equation preservation |
| `marker` | `marker-pdf>=1.10` | GPL-3.0 / Open Rail-M | Highest accuracy (95.7%), optional LLM boost |
| `pymupdf4llm` | `pymupdf4llm>=0.0.17` | AGPL | 10-50x faster, CPU-only, no LaTeX |

Backends are discovered via the **registry pattern**
(`src/research_pipeline/conversion/registry.py`). Each backend class uses
`@register_backend("name")` to self-register. The CLI `--backend` flag and
config `conversion.backend` select the active backend at runtime.

| Property | Value |
|----------|-------|
| Input | `download/pdf/*.pdf` |
| Output | `convert/markdown/*.md`, `convert/convert_manifest.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_convert.py` |
| Registry | `src/research_pipeline/conversion/registry.py` |
| Backends | `docling_backend.py`, `marker_backend.py`, `pymupdf4llm_backend.py` |
| Model | `src/research_pipeline/models/conversion.py` (`ConvertManifestEntry`) |

### 6. Extract

Chunks Markdown documents by heading structure and token limits. Builds a BM25
index for retrieval during summarization.

| Property | Value |
|----------|-------|
| Input | `convert/markdown/*.md` |
| Output | `extract/*.extract.json` |
| Implementation | `src/research_pipeline/cli/cmd_extract.py` |
| Models | `src/research_pipeline/models/extraction.py` (`ChunkMetadata`, `MarkdownExtraction`) |

### 7. Summarize

Generates per-paper summaries with evidence citations and a cross-paper
synthesis report identifying agreements, disagreements, and open questions.

| Property | Value |
|----------|-------|
| Input | `extract/*.extract.json` |
| Output | `summarize/*.summary.json`, `summarize/synthesis.json`, `summarize/synthesis.md` |
| Implementation | `src/research_pipeline/cli/cmd_summarize.py` |
| Models | `src/research_pipeline/models/summary.py` (`PaperSummary`, `SynthesisReport`) |

## Cross-cutting concerns

### Configuration

Configuration is loaded from multiple sources with this precedence
(highest wins first):

| Priority | Source | Example |
|----------|--------|---------|
| 1 | Environment variables | `ARXIV_PAPER_PIPELINE_CONFIG` |
| 2 | TOML config file | `config.toml` |
| 3 | Built-in defaults | `src/research_pipeline/config/defaults.py` |

Configuration schemas are defined in `src/research_pipeline/config/models.py`.

### Manifest tracking

Every pipeline run produces a `run_manifest.json` that records:

| Field | Description |
|-------|-------------|
| Run ID | 12-character hex identifier |
| Timestamps | Start and end times (ISO 8601) |
| Configuration | Snapshot of active settings |
| Stage records | Per-stage start/end, status, errors |
| Artifact records | File paths with SHA-256 content hashes |

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

| Tool | CLI equivalent | Description |
|------|---------------|-------------|
| `plan_topic` | `plan` | Create query plan from topic |
| `search` | `search` | Search sources for papers |
| `screen_candidates` | `screen` | Score and filter candidates |
| `download_pdfs` | `download` | Download shortlisted PDFs |
| `convert_pdfs` | `convert` | Convert PDFs to Markdown (supports backend selection) |
| `extract_content` | `extract` | Chunk and index content |
| `summarize_papers` | `summarize` | Generate summaries |
| `run_pipeline` | `run` | Run full pipeline |
| `get_run_manifest` | `inspect` | Read run manifest |
| `convert_file` | `convert-file` | Convert single PDF (supports backend selection) |
| `list_backends` | — | List available converter backends |

## Source tree

```
src/research_pipeline/
├── cli/            # Typer CLI commands (one file per stage)
├── models/         # Pydantic domain models
├── config/         # Configuration loading and schemas
├── arxiv/          # arXiv API client, XML parser, dedup, rate limiter
├── sources/        # Multi-source search adapters
├── screening/      # BM25 heuristic scoring, LLM judge interface
├── download/       # Rate-limited PDF downloader
├── conversion/     # PDF→Markdown backends (registry + docling/marker/pymupdf4llm)
├── extraction/     # Markdown chunking and retrieval
├── summarization/  # Per-paper and cross-paper synthesis
├── pipeline/       # Orchestrator and stage sequencing
├── storage/        # Workspace management, manifests, artifacts
├── infra/          # Cache, HTTP, logging, hashing, clock, paths
└── llm/            # LLM provider interface (experimental)
```
