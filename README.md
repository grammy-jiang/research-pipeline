# arxiv-paper-pipeline

A production-grade, deterministic Python pipeline for searching, screening, downloading, converting, and summarizing papers from arXiv.

## Features

- **Modular CLI** with independent, composable stage commands
- **Idempotent & resumable** — every stage can be re-run safely
- **arXiv polite-mode** — strict rate limiting, single connection, caching
- **Deterministic tool chain** with optional LLM judgment
- **Full artifact lineage** — every run is reproducible and auditable
- **Offline-first testing** — no live arXiv calls in CI

## Installation

```bash
# With uv
uv add arxiv-paper-pipeline

# With pip
pip install arxiv-paper-pipeline

# With PDF conversion support (Docling)
pip install 'arxiv-paper-pipeline[docling]'
```

## Quick Start

```bash
# Full pipeline
arxiv-paper-pipeline run "transformer architectures for time series forecasting"

# Or run stages individually
arxiv-paper-pipeline plan "transformer architectures for time series forecasting"
arxiv-paper-pipeline search --run-id <RUN_ID>
arxiv-paper-pipeline screen --run-id <RUN_ID>
arxiv-paper-pipeline download --run-id <RUN_ID>
arxiv-paper-pipeline convert --run-id <RUN_ID>
arxiv-paper-pipeline extract --run-id <RUN_ID>
arxiv-paper-pipeline summarize --run-id <RUN_ID>

# Inspect run status
arxiv-paper-pipeline inspect --run-id <RUN_ID>
```

## Commands

| Command | Purpose |
|---|---|
| `plan` | Normalize topic → query plan |
| `search` | Execute arXiv API search |
| `screen` | Two-stage relevance filtering |
| `download` | Download shortlisted PDFs |
| `convert` | PDF → Markdown (Docling) |
| `extract` | Structured content extraction |
| `summarize` | Per-paper + cross-paper synthesis |
| `run` | End-to-end orchestration |
| `inspect` | View manifests and artifacts |

## Configuration

Copy `config.example.toml` to `config.toml` and adjust settings. Key environment variables:

```
ARXIV_PAPER_PIPELINE_CONFIG       # Config file path
ARXIV_PAPER_PIPELINE_CACHE_DIR    # Override cache directory
ARXIV_PAPER_PIPELINE_WORKSPACE    # Override workspace directory
ARXIV_PAPER_PIPELINE_DISABLE_LLM  # Force LLM off
```

## Artifact Layout

```
runs/<run_id>/
├── run_config.json
├── run_manifest.json
├── plan/query_plan.json
├── search/
│   ├── raw/*.xml
│   └── candidates.jsonl
├── screen/
│   ├── cheap_scores.jsonl
│   └── shortlist.json
├── download/
│   ├── pdf/*.pdf
│   └── download_manifest.jsonl
├── convert/
│   ├── markdown/*.md
│   └── convert_manifest.jsonl
├── extract/*.extract.json
└── summarize/
    ├── *.summary.json
    ├── synthesis.json
    └── synthesis.md
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest -xvs

# Format & lint
uv run isort . && uv run black . && uv run ruff check . --fix

# Type check
uv run mypy src/
```

## License

MIT
