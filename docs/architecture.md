# Architecture

## Overview

research-pipeline is a deterministic, stage-based pipeline for academic paper
research. It automates the workflow of finding, evaluating, downloading,
converting, and summarizing scholarly papers from multiple academic databases.

| Property | Value |
|----------|-------|
| Language | Python 3.12+ |
| Build system | uv (`uv_build` backend) |
| CLI framework | Typer |
| Data models | Pydantic v2 |
| Package import | `research_pipeline` |
| CLI command | `research-pipeline` |

## Pipeline stages

The core pipeline processes a research topic through 7 sequential stages. Each
stage is **idempotent** — it can be re-run without side effects. Five auxiliary
commands extend the pipeline with quality evaluation, citation expansion,
two-tier conversion, and incremental run management.

### Core pipeline

```
plan → search → screen → download → convert → extract → summarize
```

### Auxiliary commands

```
expand       — Citation graph expansion (Semantic Scholar)
quality      — Composite quality evaluation
convert-rough — Fast Tier 2 conversion (pymupdf4llm, all PDFs)
convert-fine  — High-quality Tier 3 conversion (selected papers)
index         — Global paper index management (incremental dedup)
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

Executes the query plan against one or more sources (arXiv, Google Scholar,
Semantic Scholar, OpenAlex, DBLP) in parallel, collecting candidate papers
with metadata (title, abstract, authors, categories, dates). Cross-source
deduplication is performed by arXiv ID, DOI, and normalized title.

| Property | Value |
|----------|-------|
| Input | `plan/query_plan.json` |
| Output | `search/candidates.jsonl` (`CandidateRecord` per line), `search/raw/*.xml` |
| Implementation | `src/research_pipeline/cli/cmd_search.py` |
| Sources | `src/research_pipeline/sources/` (see Multi-source architecture below) |
| Model | `src/research_pipeline/models/candidate.py` |

### 3. Screen

Multi-stage relevance filtering. First pass uses BM25 heuristic scoring against
the query terms. Optional SPECTER2 semantic re-ranking computes embedding
similarity between the query and candidate abstracts. Optional final pass uses
an LLM judge for deeper evaluation.

| Property | Value |
|----------|-------|
| Input | `search/candidates.jsonl` |
| Output | `screen/cheap_scores.jsonl`, `screen/shortlist.json` |
| Implementation | `src/research_pipeline/cli/cmd_screen.py` |
| Scoring | `src/research_pipeline/screening/heuristic.py` |
| Embeddings | `src/research_pipeline/screening/embedding.py` (SPECTER2) |
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
Eight backends are available — three local and five cloud/online:

**Local backends** (run on your machine):

| Backend | Package | License | Strengths |
|---------|---------|---------|----------|
| `docling` | `docling>=2.0` | MIT | Great table/equation preservation |
| `marker` | `marker-pdf>=1.10` | GPL-3.0 / Open Rail-M | Highest accuracy (95.7%), optional LLM boost |
| `pymupdf4llm` | `pymupdf4llm>=0.0.17` | AGPL | 10-50x faster, CPU-only, no LaTeX |

**Online/cloud backends** (require API credentials):

| Backend | Package | Pricing | Strengths |
|---------|---------|---------|----------|
| `mathpix` | *(uses `requests`)* | ~$0.01/page, 1K free/mo | Best LaTeX equation extraction |
| `datalab` | `datalab-python-sdk>=0.1` | $4-6/1K pages, $5 free credit | Hosted Marker, fast/balanced/accurate modes |
| `llamaparse` | `llama-cloud>=1.0` | 1K free pages/day | Good quality, 4 tiers (fast/cost-effective/agentic/agentic-plus), LlamaIndex integration |
| `mistral_ocr` | `mistralai>=1.0` | Per-token, free credits | Very fast, Mistral Document AI |
| `openai_vision` | `openai>=1.0`, `PyMuPDF>=1.24` | Per-token | GPT-4o vision, page-by-page |

Backends are discovered via the **registry pattern**
(`src/research_pipeline/conversion/registry.py`). Each backend class uses
`@register_backend("name")` to self-register. The CLI `--backend` flag and
config `conversion.backend` select the active backend at runtime.

**Multi-account rotation & cross-service failover**:

Each online backend supports **multiple accounts** via `[[conversion.<backend>.accounts]]`
TOML array-of-tables. The `FallbackConverter` (`src/research_pipeline/conversion/fallback.py`)
wraps multiple backend instances and tries them in order:

1. All accounts of the primary backend are tried first.
2. If all primary accounts fail (quota exceeded, rate limited, or any error),
   it falls through to the next backend in `fallback_backends`.
3. Quota/rate-limit errors are detected via regex patterns (429, "rate limit",
   "quota exceeded", "insufficient credits", etc.).
4. Returns the first successful result, or the last failure if all backends
   are exhausted.

The `fallback_backends` config option specifies an ordered list of backup
backends to try after the primary backend:

```toml
[conversion]
backend = "mathpix"
fallback_backends = ["datalab", "mistral_ocr"]
```

| Property | Value |
|----------|-------|
| Input | `download/pdf/*.pdf` |
| Output | `convert/markdown/*.md`, `convert/convert_manifest.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_convert.py` |
| Registry | `src/research_pipeline/conversion/registry.py` |
| Fallback | `src/research_pipeline/conversion/fallback.py` (`FallbackConverter`) |
| Config models | `src/research_pipeline/config/models.py` (per-backend `*Account` + `*Config`) |
| Backends (local) | `docling_backend.py`, `marker_backend.py`, `pymupdf4llm_backend.py` |
| Backends (online) | `mathpix_backend.py`, `datalab_backend.py`, `llamaparse_backend.py`, `mistral_ocr_backend.py`, `openai_vision_backend.py` |
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

### Auxiliary: Expand (Citation Graph)

Discovers related papers by traversing the citation graph via the Semantic
Scholar API. Supports forward citations (papers that cite a given paper),
backward citations (references of a given paper), or both.

| Property | Value |
|----------|-------|
| Input | Paper IDs (arXiv IDs or S2 corpus IDs) |
| Output | `expand/expanded_candidates.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_expand.py` |
| Client | `src/research_pipeline/sources/citation_graph.py` (`CitationGraphClient`) |
| Options | `--direction` (forward/backward/both), `--limit` per direction |

### Auxiliary: Quality Evaluation

Computes a composite quality score for candidates using four weighted
dimensions: citation impact (log-normalized), venue reputation (CORE rankings),
author credibility (h-index via S2 API), and recency bonus.

| Property | Value |
|----------|-------|
| Input | `search/candidates.jsonl` |
| Output | `quality/quality_scores.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_quality.py` |
| Scoring | `src/research_pipeline/quality/` (`citation_metrics`, `venue_scoring`, `author_metrics`, `composite`) |
| Model | `src/research_pipeline/models/quality.py` (`QualityScore`) |
| Data | `src/research_pipeline/quality/data/core_rankings.json` (120+ venues) |

### Auxiliary: Convert Rough (Tier 2)

Fast bulk conversion of all downloaded PDFs using pymupdf4llm. Produces
"good enough" Markdown for initial screening and triage. Runs in parallel
with configurable worker count.

| Property | Value |
|----------|-------|
| Input | `download/pdf/*.pdf` |
| Output | `convert_rough/markdown/*.md`, `convert_rough/convert_manifest.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_convert_rough.py` |
| Backend | pymupdf4llm (always) |

### Auxiliary: Convert Fine (Tier 3)

High-quality conversion of selected papers using the configured primary
backend (Docling, Marker, or cloud). Used after initial triage to produce
publication-quality Markdown for the best papers.

| Property | Value |
|----------|-------|
| Input | `download/pdf/*.pdf` (subset via `--paper-ids`) |
| Output | `convert_fine/markdown/*.md`, `convert_fine/convert_manifest.jsonl` |
| Implementation | `src/research_pipeline/cli/cmd_convert_fine.py` |
| Backend | Configurable (primary backend from config) |

### Auxiliary: Index (Incremental Runs)

Manages a SQLite-backed global paper index that tracks papers across runs.
Enables incremental dedup — subsequent pipeline runs skip papers already
processed. Supports listing indexed papers and garbage collection.

| Property | Value |
|----------|-------|
| Input | N/A (manages index database) |
| Output | Terminal output (list/gc operations) |
| Implementation | `src/research_pipeline/cli/cmd_index.py` |
| Storage | `src/research_pipeline/storage/global_index.py` (`GlobalPaperIndex`) |
| Config | `[incremental]` section in config.toml |

## Cross-cutting concerns

### Retry & error recovery

All HTTP-based operations (PDF downloads, API calls) use the `@retry` decorator
(`src/research_pipeline/infra/retry.py`) with exponential backoff, jitter, and
`Retry-After` header support. Download and conversion manifests track
`retry_count` and `last_error` per file for debugging.

### Generic rate limiting

A reusable `RateLimiter` class (`src/research_pipeline/infra/rate_limit.py`)
provides thread-safe, monotonic-clock-based rate limiting. The arXiv-specific
`ArxivRateLimiter` extends it with a 3-second hard floor. New sources
(Semantic Scholar, OpenAlex, DBLP) each use their own `RateLimiter` instance
with configurable intervals.

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
├── ArxivSource              — arXiv API (Atom XML)
├── ScholarlySource          — Google Scholar (free, via scholarly library)
├── SerpAPISource            — Google Scholar (paid, via SerpAPI)
├── SemanticScholarSource    — Semantic Scholar API (paper search)
├── OpenAlexSource           — OpenAlex API (works endpoint)
└── DBLPSource               — DBLP API (publication search)
```

Sources run in parallel with cross-source deduplication by arXiv ID, DOI,
and normalized title. The `enrichment` module fills missing abstracts via
Semantic Scholar DOI lookup.

### CandidateRecord extensions

The `CandidateRecord` model includes optional fields for multi-source metadata:

| Field | Type | Source |
|-------|------|--------|
| `source` | `str` | All (arXiv, scholar, semantic_scholar, openalex, dblp) |
| `doi` | `str` | S2, OpenAlex, DBLP |
| `semantic_scholar_id` | `str` | Semantic Scholar |
| `openalex_id` | `str` | OpenAlex |
| `citation_count` | `int` | S2, OpenAlex |
| `influential_citation_count` | `int` | Semantic Scholar |
| `venue` | `str` | S2, DBLP |
| `year` | `int` | All |

## MCP server

The MCP server (`mcp_server/`) wraps pipeline functionality into tools
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
├── models/         # Pydantic domain models (candidate, screening, quality, etc.)
├── config/         # Configuration loading and schemas
├── arxiv/          # arXiv API client, XML parser, dedup, rate limiter
├── sources/        # Multi-source search adapters + citation graph + enrichment
├── screening/      # BM25 heuristic scoring, SPECTER2 embeddings, LLM judge
├── download/       # Rate-limited PDF downloader with retry
├── conversion/     # PDF→Markdown backends (registry + 3 local + 5 cloud + fallback)
├── quality/        # Quality evaluation (citations, venue, author, composite)
├── extraction/     # Markdown chunking and retrieval
├── summarization/  # Per-paper and cross-paper synthesis
├── pipeline/       # Orchestrator and stage sequencing
├── storage/        # Workspace management, manifests, artifacts, global index
├── infra/          # Cache, HTTP, logging, hashing, clock, rate limiting, retry
└── llm/            # LLM provider interface (experimental)
```
