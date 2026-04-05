# research-pipeline

[![PyPI version](https://img.shields.io/pypi/v/research-pipeline.svg)](https://pypi.org/project/research-pipeline/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/research-pipeline.svg)](https://pypi.org/project/research-pipeline/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, deterministic Python pipeline for searching, screening,
downloading, converting, and summarizing academic papers from arXiv and Google
Scholar.

## Features

- **7-stage pipeline**: plan → search → screen → download → convert → extract → summarize
- **Modular CLI** with independent, composable stage commands
- **MCP server** for AI agent integration (10 tools via stdio transport)
- **Multi-source search**: arXiv API + Google Scholar (free & SerpAPI)
- **Idempotent & resumable** — every stage can be re-run safely
- **arXiv polite-mode** — strict rate limiting, single connection, caching
- **Deterministic tool chain** with optional LLM judgment
- **Full artifact lineage** — every run is reproducible and auditable via manifests
- **Offline-first testing** — no live API calls in CI

## Installation

```bash
# From PyPI
pip install research-pipeline

# With PDF conversion support (Docling)
pip install research-pipeline[docling]

# With Google Scholar support
pip install research-pipeline[scholar]

# With all extras
pip install research-pipeline[docling,scholar]
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
```

## Commands

| Command | Purpose |
|---|---|
| `plan` | Normalize topic → structured query plan |
| `search` | Execute multi-source search (arXiv + Scholar) |
| `screen` | Two-stage relevance filtering (BM25 + optional LLM) |
| `download` | Download shortlisted PDFs with rate limiting |
| `convert` | PDF → Markdown via Docling |
| `extract` | Structured content extraction & chunking |
| `summarize` | Per-paper summaries + cross-paper synthesis |
| `run` | End-to-end orchestration of all stages |
| `inspect` | View run manifests and artifacts |
| `convert-file` | Standalone PDF → Markdown conversion |

## MCP server

The MCP server exposes all pipeline stages as tools for AI agent integration:

```bash
# Run via module
uv run python -m mcp_server

# Available tools: plan_topic, search, screen_candidates, download_pdfs,
# convert_pdfs, extract_content, summarize_papers, run_pipeline,
# get_run_manifest, convert_file
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
├── extract/*.extract.json     # Chunked & indexed extraction
├── summarize/
│   ├── *.summary.json         # Per-paper summaries
│   ├── synthesis.json         # Cross-paper synthesis
│   └── synthesis.md           # Human-readable synthesis
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
