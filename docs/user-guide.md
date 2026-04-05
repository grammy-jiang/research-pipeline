# User guide

## Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| Python | 3.12+ |
| [uv](https://docs.astral.sh/uv/) | latest |

## Installation

### From PyPI

```bash
# Install the base package
pip install research-pipeline

# With PDF conversion support (Docling — MIT license)
pip install research-pipeline[docling]

# With Marker backend (highest accuracy — GPL-3.0)
pip install research-pipeline[marker]

# With PyMuPDF4LLM backend (fastest — AGPL)
pip install research-pipeline[pymupdf4llm]

# With Google Scholar support
pip install research-pipeline[scholar]

# With all extras
pip install research-pipeline[docling,marker,pymupdf4llm,scholar]
```

### From source (development)

```bash
git clone https://github.com/grammy-jiang/research-pipeline.git
cd research-pipeline

# Install with all extras
uv sync --extra dev --extra docling --extra scholar
```

### Optional extras

| Extra | Purpose |
|---|---|
| `dev` | Development tools (pytest, black, ruff, mypy, pre-commit) |
| `docling` | PDF → Markdown conversion via Docling (MIT) |
| `marker` | PDF → Markdown conversion via Marker (GPL-3.0, highest accuracy) |
| `pymupdf4llm` | PDF → Markdown conversion via PyMuPDF4LLM (AGPL, fastest) |
| `scholar` | Google Scholar search via the scholarly library |
| `serpapi` | Google Scholar search via SerpAPI (requires API key) |

## Configuration

Copy the example config and customize:

```bash
cp config.example.toml config.toml
```

### Configuration file (`config.toml`)

```toml
[arxiv]
min_interval_seconds = 5.0      # Delay between arXiv API requests
default_page_size = 100         # Results per API call
daily_query_cache = true        # Cache API responses

[search]
primary_months = 6              # Search window (recent papers)
fallback_months = 12            # Fallback if too few results
max_query_variants = 5          # Query plan variations
min_candidates = 40             # Minimum candidate threshold

[screen]
cheap_top_k = 50                # Papers kept after heuristic pass
download_top_n = 8              # Papers selected for download
final_score_threshold = 0.70    # Minimum heuristic score

[download]
max_per_run = 20                # Maximum PDFs per run

[conversion]
backend = "docling"             # PDF conversion backend (docling, marker, pymupdf4llm)
timeout_seconds = 300           # Per-file timeout (docling)

[conversion.marker]             # Marker-specific settings
force_ocr = false               # Force OCR even for text PDFs
use_llm = false                 # Enable LLM-assisted conversion
llm_service = ""                # LLM service (e.g. "marker.v2")
llm_api_key = ""                # API key for LLM service

[llm]
enabled = false                 # LLM-based features (experimental)
```

### Environment variables

Environment variables override config file settings:

| Variable | Purpose |
|---|---|
| `ARXIV_PAPER_PIPELINE_CONFIG` | Path to config file |
| `ARXIV_PAPER_PIPELINE_CACHE_DIR` | Cache directory override |
| `ARXIV_PAPER_PIPELINE_WORKSPACE` | Workspace root override |
| `ARXIV_PAPER_PIPELINE_DISABLE_LLM` | Disable LLM features |

## Usage

### End-to-end pipeline

Run all stages in sequence:

```bash
research-pipeline run "transformer architectures for time series forecasting"
```

This creates a run directory at `runs/<run_id>/` and executes all 7 stages.

### Stage-by-stage execution

Each stage can be run independently, allowing inspection and adjustment between
stages:

```bash
# 1. Create a structured query plan from a research topic
research-pipeline plan "local memory systems for AI agents"
# Output: runs/<run_id>/plan/query_plan.json

# 2. Search arXiv and/or Google Scholar
research-pipeline search --run-id <RUN_ID>
# Or search with a specific source
research-pipeline search --run-id <RUN_ID> --source all
# Output: runs/<run_id>/search/candidates.jsonl

# 3. Screen candidates by relevance
research-pipeline screen --run-id <RUN_ID>
# Output: runs/<run_id>/screen/shortlist.json

# 4. Download shortlisted PDFs
research-pipeline download --run-id <RUN_ID>
# Output: runs/<run_id>/download/pdf/*.pdf

# 5. Convert PDFs to Markdown
research-pipeline convert --run-id <RUN_ID>
# Or use a specific backend
research-pipeline convert --run-id <RUN_ID> --backend marker
# Output: runs/<run_id>/convert/markdown/*.md

# 6. Extract and chunk content
research-pipeline extract --run-id <RUN_ID>
# Output: runs/<run_id>/extract/*.extract.json

# 7. Generate summaries and synthesis
research-pipeline summarize --run-id <RUN_ID>
# Output: runs/<run_id>/summarize/synthesis.md
```

### Inspecting runs

View the status and metadata of a pipeline run:

```bash
research-pipeline inspect --run-id <RUN_ID>
```

### Standalone PDF conversion

Convert a single PDF to Markdown without creating a workspace or run:

```bash
research-pipeline convert-file paper.pdf -o paper.md

# Use a specific backend
research-pipeline convert-file paper.pdf -o paper.md --backend marker
```

### Common options

All commands accept:

| Option | Short | Description |
|---|---|---|
| `--verbose` | `-v` | Enable debug logging |
| `--config` | `-c` | Path to config TOML file |
| `--workspace` | `-w` | Workspace root directory |
| `--run-id` | | Specify or resume a run ID |

## MCP server

The MCP server exposes pipeline functionality as tools for AI agent integration
via the Model Context Protocol:

```bash
uv run python -m mcp_server
```

### Available tools

| Tool | Description |
|---|---|
| `plan_topic` | Create a query plan from a topic |
| `search` | Search arXiv and Google Scholar |
| `screen_candidates` | Screen papers by relevance |
| `download_pdfs` | Download shortlisted papers |
| `convert_pdfs` | Convert PDFs to Markdown (supports backend selection) |
| `extract_content` | Chunk and extract content |
| `summarize_papers` | Generate summaries |
| `run_pipeline` | Run the full pipeline |
| `get_run_manifest` | Inspect a run's manifest |
| `convert_file` | Convert a single PDF file (supports backend selection) |
| `list_backends` | List available converter backends |

### MCP client configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "research-pipeline": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server"],
      "cwd": "/path/to/research-pipeline"
    }
  }
}
```

## Run artifacts

Each pipeline run creates a self-contained directory at `runs/<run_id>/` with
the following structure:

```
runs/<run_id>/
├── run_config.json            # Configuration snapshot for this run
├── run_manifest.json          # Stage records, artifact hashes, timestamps
├── plan/
│   └── query_plan.json        # Structured query plan
├── search/
│   ├── raw/*.xml              # Raw API responses
│   └── candidates.jsonl       # Deduplicated candidate list
├── screen/
│   ├── cheap_scores.jsonl     # BM25 heuristic scores
│   └── shortlist.json         # Final paper selection
├── download/
│   ├── pdf/*.pdf              # Downloaded papers
│   └── download_manifest.jsonl
├── convert/
│   ├── markdown/*.md          # Converted documents
│   └── convert_manifest.jsonl
├── extract/
│   └── *.extract.json         # Chunked content with BM25 index
├── summarize/
│   ├── *.summary.json         # Per-paper summaries
│   ├── synthesis.json         # Machine-readable cross-paper synthesis
│   └── synthesis.md           # Human-readable synthesis report
└── logs/
    └── pipeline.jsonl         # Structured execution logs
```

## Troubleshooting

### arXiv rate limiting

If you see HTTP 429 errors, the pipeline is being rate-limited by arXiv.
Increase `min_interval_seconds` in your config:

```toml
[arxiv]
min_interval_seconds = 10.0
```

### Docling conversion fails

Ensure the `docling` extra is installed:

```bash
uv sync --extra docling
```

Some complex PDFs may hit the timeout. Increase it:

```toml
[conversion]
timeout_seconds = 600
```

### Marker conversion fails

Ensure the `marker` extra is installed:

```bash
uv sync --extra marker
```

Marker requires PyTorch. On first use it downloads model weights (~1 GB).

### PyMuPDF4LLM conversion

The fastest backend but does not render LaTeX equations:

```bash
uv sync --extra pymupdf4llm
```

### Google Scholar access

The free `scholarly` library may be blocked by Google. For reliable access,
use SerpAPI:

```bash
uv sync --extra serpapi
```

Set your API key in the config or environment.
