# research-pipeline

[![PyPI version](https://img.shields.io/pypi/v/research-pipeline.svg)](https://pypi.org/project/research-pipeline/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/research-pipeline.svg)](https://pypi.org/project/research-pipeline/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, deterministic Python pipeline for searching, screening,
downloading, converting, and summarizing academic papers from arXiv, Google
Scholar, Semantic Scholar, OpenAlex, and DBLP.

## Features

- **Multi-stage pipeline**: plan → search → screen → download → convert → extract → summarize
- **5 new auxiliary commands**: `expand` (citation graph), `quality` (evaluation scoring), `convert-rough` / `convert-fine` (two-tier conversion), `index` (incremental runs)
- **Modular CLI** with independent, composable stage commands
- **MCP server** for AI agent integration (17 tools, 15 resources, 6 prompts, completions, progress reporting)
- **Harness-engineered research workflow** — server-driven orchestration with 6 layers: telemetry, context engineering, governance, structural verification, doom-loop monitoring, and crash recovery
- **Multi-source search**: arXiv + Google Scholar + Semantic Scholar + OpenAlex + DBLP
- **Cross-source enrichment** — fill missing abstracts via DOI lookup
- **Semantic re-ranking** — optional SPECTER2 embeddings for similarity scoring
- **Citation graph expansion** — discover related papers via Semantic Scholar citations
- **Quality evaluation** — composite scoring: citation impact, venue reputation, author h-index, recency
- **Multi-backend PDF conversion**: 3 local (Docling, Marker, PyMuPDF4LLM) + 5 cloud (Mathpix, Datalab, LlamaParse, Mistral OCR, OpenAI Vision)
- **Two-tier conversion** — fast `convert-rough` for all papers, high-quality `convert-fine` for selected ones
- **Multi-account rotation** — rotate between accounts per service on quota exhaustion
- **Cross-service fallback** — automatic failover to next backend when all accounts are exhausted
- **Incremental runs** — SQLite global index deduplicates papers across runs
- **Retry & error recovery** — `@retry` decorator with exponential backoff, jitter, and Retry-After support
- **Idempotent & resumable** — every stage can be re-run safely
- **arXiv polite-mode** — strict rate limiting, single connection, caching
- **Deterministic tool chain** with optional LLM judgment
- **Full artifact lineage** — every run is reproducible and auditable via manifests
- **Offline-first testing** — no live API calls in CI

## Installation

```bash
# From PyPI
pip install research-pipeline

# With local PDF conversion backends
pip install research-pipeline[docling]       # MIT license, great tables/equations
pip install research-pipeline[marker]        # Highest accuracy (95.7%), GPL-3.0
pip install research-pipeline[pymupdf4llm]   # Fastest (10-50x), AGPL

# With cloud PDF conversion backends (require API keys)
pip install research-pipeline[mathpix]       # Best LaTeX, 1K free pages/mo
pip install research-pipeline[datalab]       # Hosted Marker, $5 free credit
pip install research-pipeline[llamaparse]    # 1K free pages/day
pip install research-pipeline[mistral-ocr]   # Mistral OCR, free credits
pip install research-pipeline[openai-vision] # GPT-4o vision

# With Google Scholar support
pip install research-pipeline[scholar]

# With all extras
pip install research-pipeline[docling,marker,pymupdf4llm,scholar]
```

### Development install

```bash
# With uv (recommended)
uv sync --extra dev --extra docling --extra scholar
```

## Quick start

```bash
# Full end-to-end pipeline
research-pipeline run "transformer architectures for time series forecasting"

# Or run stages individually
research-pipeline plan "transformer architectures for time series forecasting"
research-pipeline search --run-id <RUN_ID>
research-pipeline screen --run-id <RUN_ID>
research-pipeline download --run-id <RUN_ID>
research-pipeline convert --run-id <RUN_ID>
research-pipeline extract --run-id <RUN_ID>
research-pipeline summarize --run-id <RUN_ID>

# Inspect run status
research-pipeline inspect --run-id <RUN_ID>

# Standalone PDF conversion (no workspace required)
research-pipeline convert-file paper.pdf -o paper.md

# Use a specific conversion backend
research-pipeline convert --run-id <RUN_ID> --backend marker
research-pipeline convert-file paper.pdf --backend pymupdf4llm

# Two-tier conversion: rough (fast) then fine (high-quality)
research-pipeline convert-rough --run-id <RUN_ID>
research-pipeline convert-fine --run-id <RUN_ID>

# Evaluate paper quality (citation impact, venue, author)
research-pipeline quality --run-id <RUN_ID>

# Expand via citation graph (Semantic Scholar)
research-pipeline expand --run-id <RUN_ID> --direction both

# Manage global paper index (incremental dedup)
research-pipeline index --list
```

## Commands

| Command | Purpose |
|---|---|
| `plan` | Normalize topic → structured query plan |
| `search` | Execute multi-source search (arXiv, Scholar, Semantic Scholar, OpenAlex, DBLP) |
| `screen` | Two-stage relevance filtering (BM25 + optional SPECTER2 + optional LLM) |
| `download` | Download shortlisted PDFs with rate limiting and retry |
| `convert` | PDF → Markdown (8 backends, multi-account rotation, cross-service fallback) |
| `convert-rough` | Fast Tier 2 conversion (pymupdf4llm) for all downloaded PDFs |
| `convert-fine` | High-quality Tier 3 conversion for selected papers |
| `extract` | Structured content extraction & chunking |
| `summarize` | Per-paper summaries + cross-paper synthesis |
| `expand` | Citation graph expansion via Semantic Scholar API |
| `quality` | Composite quality evaluation (citations, venue, author, recency) |
| `run` | End-to-end orchestration of all stages |
| `inspect` | View run manifests and artifacts |
| `convert-file` | Standalone PDF → Markdown conversion |
| `index` | Manage the global paper index for incremental runs |
| `install-skill` | Install the Claude/Copilot skill to `~/.claude/skills/` |

## MCP server

The MCP server provides full Model Context Protocol support for AI agent
integration:

```bash
# Run via module
uv run python -m mcp_server
```

**17 tools** — all pipeline stages plus auxiliary commands and workflow:

`plan_topic`, `search`, `screen_candidates`, `download_pdfs`, `convert_pdfs`,
`extract_content`, `summarize_papers`, `run_pipeline`, `get_run_manifest`,
`convert_file`, `list_backends`, `expand_citations`, `evaluate_quality`,
`convert_rough`, `convert_fine`, `manage_index`, `research_workflow`

**15 resources** — read pipeline artifacts via URI templates:

`runs://list`, `runs://{run_id}/manifest`, `runs://{run_id}/plan`,
`runs://{run_id}/candidates`, `runs://{run_id}/shortlist`,
`runs://{run_id}/papers/{paper_id}`, `runs://{run_id}/markdown/{paper_id}`,
`runs://{run_id}/summary/{paper_id}`, `runs://{run_id}/synthesis`,
`runs://{run_id}/quality`, `config://current`, `index://papers`,
`workflow://{run_id}/state`, `workflow://{run_id}/telemetry`,
`workflow://{run_id}/budget`

**6 prompts** — research workflow templates:

`research_topic`, `research_workflow`, `analyze_paper`, `compare_papers`,
`refine_search`, `quality_assessment`

Plus: **tool annotations**, **auto-completions**, and **progress reporting**.

### Harness-engineered workflow

The `research_workflow` tool drives a server-side orchestrated research workflow
with 6 harness engineering layers derived from a 79-paper synthesis:

| Layer | Purpose |
|-------|---------|
| WL1 Telemetry | Three-surface logging (cognitive/operational/contextual) |
| WL2 Context | Token budgets, 5-stage paper compaction (Tokalator/ACC) |
| WL3 Governance | Schema-level state machine, verify-before-commit gates |
| WL4 Verification | Structural output validation (not self-referential) |
| WL5 Monitoring | Doom-loop detection, iteration drift tracking |
| WL6 Recovery | Persistent state after every stage, crash-recovery |

Features:
- **Sampling-based analysis**: LLM paper analysis via `create_message()` (1 round per paper)
- **Elicitation gates**: user approval at 6 decision points via `ctx.elicit()`
- **Iterative synthesis**: system-building mode with gap analysis and convergence
- **Bounded rationality**: max 3 iterations, 7 explicit stop conditions
- **Graceful degradation**: works without sampling or elicitation capabilities

## AI skill

Install the bundled Claude Code / GitHub Copilot skill:

```bash
# Copy skill files to ~/.claude/skills/research-pipeline/
research-pipeline install-skill

# Or create a symlink (for development)
research-pipeline install-skill --symlink

# Force overwrite existing
research-pipeline install-skill --force
```

## Configuration

Copy `config.example.toml` to `config.toml` and adjust settings:

```bash
cp config.example.toml config.toml
```

Key environment variables:

| Variable | Purpose |
|---|---|
| `ARXIV_PAPER_PIPELINE_CONFIG` | Config file path |
| `ARXIV_PAPER_PIPELINE_CACHE_DIR` | Override cache directory |
| `ARXIV_PAPER_PIPELINE_WORKSPACE` | Override workspace directory |
| `ARXIV_PAPER_PIPELINE_DISABLE_LLM` | Force LLM off |

## Artifact layout

Each pipeline run produces outputs in `runs/<run_id>/`:

```
runs/<run_id>/
├── run_config.json            # Configuration snapshot
├── run_manifest.json          # Execution metadata & stage records
├── plan/query_plan.json       # Normalized query plan
├── search/
│   ├── raw/*.xml              # Raw API response pages
│   └── candidates.jsonl       # Deduplicated candidates
├── screen/
│   ├── cheap_scores.jsonl     # Heuristic scores
│   └── shortlist.json         # Papers selected for download
├── download/
│   ├── pdf/*.pdf              # Downloaded papers
│   └── download_manifest.jsonl
├── convert/
│   ├── markdown/*.md          # Converted Markdown
│   └── convert_manifest.jsonl
├── convert_rough/             # Tier 2: fast conversion (all PDFs)
│   ├── markdown/*.md
│   └── convert_manifest.jsonl
├── convert_fine/              # Tier 3: high-quality conversion (selected)
│   ├── markdown/*.md
│   └── convert_manifest.jsonl
├── quality/                   # Quality evaluation scores
│   └── quality_scores.jsonl
├── expand/                    # Citation graph expansion
│   └── expanded_candidates.jsonl
├── extract/*.extract.json     # Chunked & indexed extraction
├── summarize/
│   ├── *.summary.json         # Per-paper summaries
│   ├── synthesis.json         # Cross-paper synthesis
│   └── synthesis.md           # Human-readable synthesis
├── workflow/                  # Harness-engineered workflow state
│   ├── state.json             # Workflow state (stage statuses, execution log)
│   └── telemetry.jsonl        # Three-surface telemetry events
└── logs/pipeline.jsonl        # Structured logs
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/unit/ -xvs

# Format, lint, type check
uv run isort . && uv run black . && uv run ruff check . --fix
uv run mypy src/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture
documentation and [docs/user-guide.md](docs/user-guide.md) for the full user
guide.

## License

MIT
