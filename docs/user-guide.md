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

# With MinerU backend (TEDS 93.42% tables — MIT)
pip install 'research-pipeline[mineru]'

# With online/cloud backends
pip install research-pipeline[mathpix]       # Mathpix (best LaTeX, 1K free/mo)
pip install research-pipeline[datalab]       # Datalab (hosted Marker, $5 free credit)
pip install research-pipeline[llamaparse]    # LlamaParse (1K free pages/day)
pip install research-pipeline[mistral-ocr]   # Mistral OCR (per-token, free credits)
pip install research-pipeline[openai-vision] # OpenAI GPT-4o vision (per-token)

# With Google Scholar support
pip install research-pipeline[scholar]

# With cross-encoder reranking for higher-precision chunk retrieval
pip install research-pipeline[reranker]

# With all extras
pip install research-pipeline[docling,marker,pymupdf4llm,scholar,reranker]
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
| `mineru` | PDF → Markdown conversion via MinerU / magic-pdf (MIT, TEDS 93.42%) |
| `mathpix` | Mathpix cloud OCR (best LaTeX, 1K free pages/mo) |
| `datalab` | Datalab hosted Marker ($5 free credit) |
| `llamaparse` | LlamaParse cloud parsing (1K free pages/day) |
| `mistral-ocr` | Mistral Document AI OCR (per-token, free credits) |
| `openai-vision` | OpenAI GPT-4o vision (per-token) |
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
use_semantic_reranking = false  # Enable SPECTER2 semantic re-ranking
embedding_model = "allenai/specter2"  # HuggingFace embedding model
embedding_batch_size = 32       # Batch size for embeddings

[download]
max_per_run = 20                # Maximum PDFs per run

[conversion]
backend = "docling"             # Backend: docling, marker, pymupdf4llm, mineru, mathpix,
                                #   datalab, llamaparse, mistral_ocr, openai_vision
fallback_backends = []          # Ordered list of backup backends (see below)
timeout_seconds = 300           # Per-file timeout (docling)
rough_max_workers = 4           # Parallel workers for convert-rough
fine_max_workers = 2            # Parallel workers for convert-fine

[conversion.marker]             # Marker-specific settings
force_ocr = false               # Force OCR even for text PDFs
use_llm = false                 # Enable LLM-assisted conversion
llm_service = ""                # LLM service (e.g. "marker.v2")
llm_api_key = ""                # API key for LLM service

[conversion.mathpix]            # Mathpix cloud OCR
app_id = ""                     # RESEARCH_PIPELINE_MATHPIX_APP_ID
app_key = ""                    # RESEARCH_PIPELINE_MATHPIX_APP_KEY

[conversion.datalab]            # Datalab (hosted Marker)
api_key = ""                    # RESEARCH_PIPELINE_DATALAB_API_KEY
mode = "balanced"               # fast, balanced, or accurate

[conversion.llamaparse]         # LlamaParse
api_key = ""                    # RESEARCH_PIPELINE_LLAMAPARSE_API_KEY
tier = "agentic"                # fast (1 credit), cost-effective (3), agentic (10), agentic-plus (45)

[conversion.mistral_ocr]        # Mistral OCR
api_key = ""                    # RESEARCH_PIPELINE_MISTRAL_API_KEY
model = "mistral-ocr-latest"

[conversion.mineru]             # MinerU (magic-pdf) — TEDS 93.42% tables
parse_method = "auto"           # "auto", "ocr", or "txt"
timeout_seconds = 600

[conversion.openai_vision]      # OpenAI GPT-4o vision
api_key = ""                    # RESEARCH_PIPELINE_OPENAI_API_KEY
model = "gpt-4o"

[llm]
enabled = false                 # LLM-based features (experimental)
provider = "ollama"             # Provider: "ollama" or "openai"
base_url = ""                   # Provider URL (empty = default)
api_key = ""                    # API key (required for OpenAI)
model = ""                      # Model name (empty = provider default)
max_tokens = 4096               # Max output tokens

[llm_routing]
enabled = false                 # Phase-aware model routing (v0.13.6+)

[llm_routing.mechanical]        # Cheap/local model for mechanical stages
provider = "ollama"             # (plan, search, download, convert, extract, index)
model = ""

[llm_routing.intelligent]       # Capable model for analysis stages
provider = "openai"             # (screen, summarize, expand, quality, analyze, aggregate)
model = "gpt-4o"
api_key = ""

[llm_routing.critical_safety]   # Premium model for safety-critical stages
provider = "openai"             # (validate, compare, security_gate)
model = "gpt-4o"
api_key = ""

[gates]
enabled = false                 # Human-in-the-loop gates (v0.13.7+)
auto_approve = true             # Auto-approve all gates (set false for interactive)
gate_after = ["screen", "download", "summarize"]  # Stages after which to pause

# Multi-session coherence evaluation (v0.13.8+)
# CLI: research-pipeline coherence <RUN_A> <RUN_B> [<RUN_C> ...]
# Evaluates factual consistency, temporal ordering, knowledge update fidelity,
# and contradiction detection across 2+ pipeline runs.

# Memory consolidation (v0.13.9+)
# CLI: research-pipeline consolidate [RUN_IDS...] [--dry-run] [--capacity N]
# Episodic→semantic consolidation: compresses old episodes into rules,
# prunes stale entries, tracks semantic drift between runs.
# Options: --capacity (default 100), --threshold (0.8), --min-support (2),
#          --staleness-days (90), --dry-run, --output path

# Epistemic blinding audits (v0.13.10+)
# CLI: research-pipeline blinding-audit [--run-id ID] [--threshold 0.4]
# A/B blinding protocol for detecting LLM prior contamination in analysis.
# Scans findings for author/title/venue/year/citation references.
# Score < 0.2 = LOW, 0.2-0.4 = MEDIUM, > 0.4 = HIGH contamination.
# Options: --threshold (0.4), --no-store, --json, --workspace

[sources]
default_sources = ["arxiv"]     # Sources: arxiv, scholar, semantic_scholar, openalex, dblp
semantic_scholar_api_key = ""   # S2 API key (optional, higher rate limits)
semantic_scholar_min_interval = 1.0
openalex_min_interval = 0.1
openalex_email = ""             # Polite pool email
dblp_min_interval = 1.0

[quality]
enabled = false                 # Enable quality evaluation
citation_weight = 0.35          # Weight for citation impact
venue_weight = 0.25             # Weight for venue reputation
author_weight = 0.25            # Weight for author h-index
recency_weight = 0.15           # Weight for recency bonus

[incremental]
enabled = false                 # Enable incremental dedup across runs
global_index_path = ""          # SQLite path (empty = default cache dir)
reuse_artifacts = true          # Symlink existing PDFs/markdown
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

This creates a run directory at `runs/<run_id>/` and executes all 7 core stages.

### Pipeline profiles

Control execution depth with `--profile` (v0.12.6+):

```bash
# Quick: abstract-only overview (no PDF download/conversion)
research-pipeline run --profile quick "transformer attention"

# Standard: full 7-stage pipeline (default)
research-pipeline run "local memory systems for AI agents"

# Deep: standard + citation expansion + quality scoring + claim analysis
research-pipeline run --profile deep "comprehensive survey of memory in LLMs"

# Auto: detect profile from query complexity
research-pipeline run --profile auto "what is RLHF"
```

| Profile | Stages | When to use |
|---------|--------|-------------|
| `quick` | plan→search→screen→summarize | Fast overviews, simple lookups |
| `standard` | Full 7-stage pipeline | Default research workflow |
| `deep` | standard + expand + quality + claims + TER | Comprehensive surveys, comparisons |
| `auto` | Auto-detected | Let the pipeline decide |

Set the default profile in `config.toml`:

```toml
profile = "standard"  # quick, standard, deep, or auto
```

### Iterative gap-filling (TER loop)

The deep profile includes a THINK→EXECUTE→REFLECT loop that iteratively
identifies gaps in the synthesis and generates targeted queries to fill them.

```bash
# Default: 3 TER iterations
research-pipeline run --profile deep "comprehensive survey of memory in LLMs"

# More iterations for thorough gap-filling
research-pipeline run --profile deep --ter-iterations 5 "LLM survey"

# Disable TER loop
research-pipeline run --profile deep --ter-iterations 0 "quick deep run"
```

Configure in `config.toml`:

```toml
ter_max_iterations = 3  # 0 to disable
```

The loop converges automatically when:
- No gaps are found in the synthesis
- Gap count stops decreasing between iterations
- No new queries can be generated
- Maximum iterations reached

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

# 3a. Record feedback on screening results (optional, improves future runs)
research-pipeline feedback --run-id <RUN_ID> --accept 2401.12345 --reject 2401.99999
# Recompute adjusted weights: research-pipeline feedback --run-id <RUN_ID> --adjust

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
# With cross-encoder reranking (requires sentence-transformers)
research-pipeline extract --run-id <RUN_ID> --cross-encoder
# Output: runs/<run_id>/extract/*.extract.json

# 7. Generate summaries and synthesis
research-pipeline summarize --run-id <RUN_ID>
# Output: runs/<run_id>/summarize/synthesis.md

# Export in alternative formats (v0.12.0+)
research-pipeline summarize --run-id <RUN_ID> -f json       # JSON with metadata + evidence
research-pipeline summarize --run-id <RUN_ID> -f bibtex     # BibTeX bibliography
research-pipeline summarize --run-id <RUN_ID> -f structured-json  # Claim-evidence chains
```

### Claim decomposition (v0.12.3+)

After summarization, decompose findings into atomic claims with evidence
classification:

```bash
# Decompose paper summaries into atomic claims with evidence support
research-pipeline analyze-claims --run-id <RUN_ID>
# Output: runs/<run_id>/summarize/claims/claim_decomposition.jsonl
```

Each claim is classified as **supported**, **partial**, **conflicting**,
**inconclusive**, or **unsupported** based on BM25 retrieval against the
source markdown chunks.

### Confidence scoring (v0.12.5+)

After claim decomposition, compute multi-signal confidence scores per claim:

```bash
# Score claims using evidence strength, hedging detection, citation density
research-pipeline score-claims --run-id <RUN_ID>
# Output: runs/<run_id>/summarize/claims/scored_claims.jsonl
```

Signals include **evidence strength** (from evidence class), **hedging language
detection** (LVU — linguistic markers like "might", "possibly", "suggests"),
**citation density** (number of supporting evidence chunks), and **retrieval
quality** (max BM25 score). When an LLM is available, adds **multi-sample
consistency verification** (M=5 samples at temperature 0.7).

### Knowledge graph (v0.12.4+)

Build a typed knowledge graph from pipeline results:

```bash
# Ingest candidates and claims into the knowledge graph
research-pipeline kg-ingest --run-id <RUN_ID>
# Output: ~/.cache/research-pipeline/knowledge_graph.db

# View knowledge graph statistics
research-pipeline kg-stats

# Query an entity and its relations
research-pipeline kg-query 2401.12345
```

The knowledge graph stores 7 entity types (paper, concept, method, experiment,
claim, author, venue) and 10 relation types with full provenance tracking.
Use `--db` to specify a custom database path.

### Three-tier memory (v0.12.8+)

Persistent research context across pipeline stages and runs:

```bash
# View memory statistics (working, episodic, semantic)
research-pipeline memory-stats

# List recent past runs
research-pipeline memory-episodes --limit 5

# Search episodic memory for past runs on a topic
research-pipeline memory-search "transformer"
```

The three tiers:

- **Working memory**: Bounded per-stage FIFO buffer (configurable via
  `memory_working_capacity`). Automatically resets at stage boundaries.
- **Episodic memory**: SQLite-backed run history
  (`~/.cache/research-pipeline/episodic_memory.db`). Records topic, paper
  count, stages completed, and outcome for each run.
- **Semantic memory**: Cross-run knowledge via the knowledge graph. Queries
  known concepts, methods, and papers to inform new research.

The orchestrator automatically uses memory during `run`:
- Checks prior knowledge before starting (episodic + semantic)
- Resets working memory at each stage boundary
- Records an episode when the run completes

### Content security gates (v0.12.9+)

Defense-in-depth for untrusted content entering the pipeline:

- **Boundary classifiers**: 4-level risk classification (clean/low/medium/high)
  at each ingestion point. Detects prompt injection, template injection, null
  bytes, Unicode overrides, data URIs, and other suspicious patterns.
- **Taint tracking**: Each piece of content is labeled with its provenance
  (source, stage) and trust level (trusted/semi-trusted/untrusted). Taint
  propagates when content is transformed.
- **Security gates**: Combined classify → sanitize → quarantine pipeline at
  stage boundaries. Medium-risk content is sanitized; high-risk content is
  quarantined (blocked).

Security gates run automatically during `run` — no configuration required.
Stats are logged at the end of each pipeline run.

### Schema-grounded evaluation (v0.12.10+)

Automated validation of pipeline outputs against their Pydantic model schemas:

- **Field completeness**: Required fields present and non-empty
- **Score range validation**: Numeric scores in expected bounds (e.g., [0, 1])
- **Cross-field consistency**: Count fields match actual list lengths
- **Per-stage evaluators**: plan, search, screen, summarize

```bash
# Evaluate all stages of a run
research-pipeline evaluate --run-id <RUN_ID>

# Evaluate a specific stage
research-pipeline evaluate --run-id <RUN_ID> --stage plan

# Verbose output with debug info
research-pipeline evaluate --run-id <RUN_ID> -v
```

Evaluation runs automatically after each stage in the orchestrator (informational
only — failures are logged as warnings but never block pipeline execution).

### Self-improving retrieval (v0.12.11+)

After screening, the pipeline runs a Self-Improving Retrieval (SIR) loop that
iteratively refines query terms based on result feedback:

- **Term feedback**: Drops terms with no coverage, adds frequent terms from top results
- **Coverage tracking**: Measures fraction of query terms found in paper titles/abstracts
- **Convergence detection**: Stops on score plateau, term stability, or max iterations
- **No LLM required**: Works entirely with heuristic term analysis

SIR runs automatically after the screen stage and saves `sir_result.json` in the
screen directory. The result includes iteration history, convergence reason, and
the final refined query terms.

### Tiered page dispatch (v0.12.12+)

Page-level difficulty classification for intelligent PDF conversion routing:

- **Page analysis**: Inspects each page for math, tables, images, text density
- **Three tiers**: SIMPLE (text-only), MODERATE (images), COMPLEX (math/tables)
- **Backend recommendation**: pymupdf4llm → marker → docling based on complexity
- **Dispatch plan**: Saves `page_dispatch.json` per run with per-PDF analysis

Runs automatically before the convert stage. No configuration needed — the
dispatch plan is informational and helps guide backend selection.

### Multi-agent analysis (v0.12.13+)

Parallel per-paper analysis using coordinated sub-agents:

- **MasterAgent/SubAgent**: ThreadPoolExecutor-based coordination for independent analysis tasks
- **Conflict detection**: cross-sub-agent output comparison with LOW/MEDIUM/HIGH severity
- **Evidence merging**: "collect" gathers everything, "evidence_only" filters to claim/evidence fields
- **Output**: `agent_analysis.json` in the summarize directory with completion stats and conflicts

Runs automatically during the summarize stage. Non-blocking — failures log a warning
and the pipeline continues normally.

### MCP zero-trust security (v0.12.14+)

MCPSHIELD-inspired 4-layer defense for MCP tool interactions:

- **Tool pinning**: SHA-256 hash verification of tool schemas to detect tampering
- **Trust domains**: Tools classified as READ, WRITE, EXECUTE, NETWORK, or SYSTEM
- **Capability control**: Per-caller grant/revoke with deny-all override
- **Rate limiting**: Per-tool calls-per-minute enforcement
- **Audit trail**: Full invocation log with `mcp_audit.json` summary per run

Initializes automatically at pipeline start. Non-blocking — guard failures log a
warning and the pipeline continues normally.

### User feedback loop (v0.13.0+)

Record accept/reject decisions on screened papers to improve future
screening via ELO-style BM25 weight adjustment:

```bash
# Accept relevant papers
research-pipeline feedback --run-id <RUN_ID> --accept 2401.12345 --accept 2401.12346

# Reject irrelevant papers with reason
research-pipeline feedback --run-id <RUN_ID> --reject 2401.99999 --reason "off-topic"

# View feedback stats
research-pipeline feedback --run-id <RUN_ID> --show

# Recompute adjusted weights from all accumulated feedback
research-pipeline feedback --run-id <RUN_ID> --adjust
```

Feedback is stored in `~/.cache/research-pipeline/feedback.db`. After ≥5
records (with both accepts and rejects), the screen stage automatically uses
feedback-adjusted weights. The adjustment uses an ELO-inspired algorithm:
weights that correlate with accepted papers are boosted, weights that
correlate with rejected papers are dampened.

### Three-channel eval logging (v0.13.1+)

Three evaluation channels capture pipeline execution for observability
and post-hoc analysis:

1. **Execution traces** — Structured JSONL with timing and causality
   (`<run_root>/logs/traces.jsonl`)
2. **Audit database** — SQLite who/what/when records
   (`<run_root>/logs/audit.db`)
3. **Environment snapshots** — Filesystem state at stage boundaries
   (`<run_root>/snapshots/`)

Eval logging is automatic during pipeline runs. To inspect:

```bash
# View all channels
research-pipeline eval-log --run-id <RUN_ID>

# View specific channel
research-pipeline eval-log --run-id <RUN_ID> --channel traces --stage screen
research-pipeline eval-log --run-id <RUN_ID> --channel audit --limit 20
research-pipeline eval-log --run-id <RUN_ID> --channel summary
```

### Evidence-only aggregation (v0.13.2+)

Strips rhetoric from synthesis outputs, normalizes statement length,
requires evidence citations, merges duplicates, and filters unsupported
claims:

```bash
research-pipeline aggregate --run-id <RUN_ID>
research-pipeline aggregate --run-id <RUN_ID> --min-pointers 1 --format json
```

### HTML report export

Export synthesis reports as self-contained HTML with citation links,
confidence badges, dark mode support, and responsive design:

```bash
# From structured synthesis JSON
research-pipeline export-html --run-id <RUN_ID>

# From any Markdown file
research-pipeline export-html --markdown report.md -o report.html --title "My Research"
```

### Auxiliary commands

These commands extend the core pipeline with additional capabilities:

```bash
# Citation graph expansion via Semantic Scholar
research-pipeline expand --run-id <RUN_ID> --direction both --limit 20
# Output: runs/<run_id>/expand/expanded_candidates.jsonl

# BFS multi-hop expansion with BM25 pruning (+24pp recall)
research-pipeline expand --run-id <RUN_ID> --paper-ids "2401.12345" \
  --bfs-depth 2 --bfs-top-k 10 --bfs-query "transformer,attention"

# Snowball expansion with budget-aware stopping (v0.13.3+)
research-pipeline expand --run-id <RUN_ID> --paper-ids "2401.12345" \
  --snowball --bfs-query "harness,engineering" --snowball-max-rounds 5
# Output: expand/expanded_candidates.jsonl + snowball_report.md + snowball_stats.json

# Quality evaluation (citation impact, venue, author, recency, safety gate)
research-pipeline quality --run-id <RUN_ID>
# Output: runs/<run_id>/quality/quality_scores.jsonl

# Two-tier conversion: rough (fast, all PDFs) then fine (high-quality, selected)
research-pipeline convert-rough --run-id <RUN_ID>
# Output: runs/<run_id>/convert_rough/markdown/*.md
research-pipeline convert-fine --run-id <RUN_ID>
# Or convert-fine for specific papers only
research-pipeline convert-fine --run-id <RUN_ID> --paper-ids "2401.12345,2402.67890"
# Output: runs/<run_id>/convert_fine/markdown/*.md

# Manage global paper index for incremental dedup
research-pipeline index --list
research-pipeline index --gc

# Validate a research report (structure, RACE quality, FACT citations)
research-pipeline validate --report report.md
# With FACT citation verification against a run's corpus (v0.12.0+)
research-pipeline validate --run-id <RUN_ID> --workspace runs
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

The MCP server provides full Model Context Protocol support for AI agent
integration via stdio transport:

```bash
uv run python -m mcp_server
```

### Tools (18)

All 20 tools include **annotations** (readOnlyHint, destructiveHint,
idempotentHint, openWorldHint) and **progress reporting** via MCP context.

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
| `expand_citations` | Expand candidate pool via citation graph |
| `evaluate_quality` | Compute composite quality scores |
| `convert_rough` | Fast conversion of all PDFs (pymupdf4llm) |
| `convert_fine` | High-quality conversion of selected papers |
| `manage_index` | Manage the global paper index |
| `record_feedback` | Record accept/reject feedback on screened papers |
| `research_workflow` | **Server-driven orchestrated workflow** (see below) |

### Resources (15)

Read pipeline artifacts via URI templates:

| URI | Description |
|---|---|
| `runs://list` | List all run IDs |
| `runs://{run_id}/manifest` | Run manifest (stages, timestamps, hashes) |
| `runs://{run_id}/plan` | Query plan for a run |
| `runs://{run_id}/candidates` | Search candidates (JSONL) |
| `runs://{run_id}/shortlist` | Screened shortlist |
| `runs://{run_id}/papers/{paper_id}` | Downloaded PDF metadata |
| `runs://{run_id}/markdown/{paper_id}` | Converted Markdown content |
| `runs://{run_id}/summary/{paper_id}` | Per-paper summary |
| `runs://{run_id}/synthesis` | Cross-paper synthesis report |
| `runs://{run_id}/quality` | Quality evaluation scores |
| `config://current` | Current pipeline configuration |
| `index://papers` | Global paper index |
| `workflow://{run_id}/state` | Workflow state (stage statuses, execution log) |
| `workflow://{run_id}/telemetry` | Telemetry events (JSONL) |
| `workflow://{run_id}/budget` | Context budget usage |

### Prompts (6)

Research workflow templates for AI agents:

| Prompt | Description |
|---|---|
| `research_topic` | Plan and execute a research workflow on a topic |
| `research_workflow` | Harness-engineered workflow with sampling and elicitation |
| `analyze_paper` | Deep analysis of a specific paper |
| `compare_papers` | Compare papers within a run |
| `refine_search` | Improve search results after screening |
| `quality_assessment` | Interpret quality evaluation scores |

### Research workflow tool

The `research_workflow` tool provides a server-driven orchestrated workflow
with 6 harness engineering layers:

```
Telemetry → Context → Execute → Verify → Govern → Monitor → Recovery → Telemetry
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `topic` | (required) | Research topic to investigate |
| `workspace` | `./workspace` | Workspace directory |
| `run_id` | auto | Run identifier |
| `system_building` | `false` | Enable iterative synthesis mode |
| `source` | config default | Search sources (`arxiv`, `scholar`, `all`) |
| `max_iterations` | `3` | Maximum synthesis iterations |
| `resume` | `false` | Resume from saved state |

**Capabilities:**

- **With sampling**: LLM-based paper analysis and cross-paper synthesis
- **With elicitation**: user approval at 6 decision gates (plan, shortlist,
  optional stages, conversion selection, iteration, report)
- **Without sampling**: pipeline-only mode (plan through summarize)
- **Without elicitation**: uses sensible defaults at all gates

**Harness layers:**

| Layer | Description |
|---|---|
| WL1 Telemetry | Three-surface logging (cognitive, operational, contextual) |
| WL2 Context | Token budgets with 5-stage paper compaction |
| WL3 Governance | Schema-level state machine with verify-before-commit |
| WL4 Verification | Structural output validation (not LLM-as-judge) |
| WL5 Monitoring | Doom-loop detection via MD5 fingerprinting |
| WL6 Recovery | Persistent state after every stage for crash-recovery |

### Completions

Auto-complete support for: `run_id`, `paper_id`, `backend`, `direction`,
`source`, `topic`.

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

## AI skill

The package includes a bundled skill and sub-agent definitions for Claude Code
and GitHub Copilot that provide step-by-step research workflow instructions,
configuration, reference documentation, and specialized analysis agents.

```bash
# Install skill + agents to ~/.claude/
research-pipeline setup

# Create symlinks (useful during development)
research-pipeline setup --symlink

# Force overwrite existing
research-pipeline setup --force

# Install only the skill (skip agents)
research-pipeline setup --skip-agents

# Install only agents (skip skill)
research-pipeline setup --skip-skill
```

This installs:
- **Skill** → `~/.claude/skills/research-pipeline/` — SKILL.md with workflow
  instructions, configuration template, and reference docs
- **Sub-agents** → `~/.claude/agents/` — paper-analyzer.md, paper-screener.md,
  paper-synthesizer.md for deep paper analysis, screening, and synthesis

Once installed, Claude Code and GitHub Copilot will automatically discover the
skill and agents when you ask about academic paper research.

> **Note:** The deprecated `install-skill` command still works as a hidden alias
> for `setup` but may be removed in a future release.

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
│   ├── synthesis.md           # Human-readable synthesis report
│   └── claims/
│       ├── claim_decomposition.jsonl  # Atomic claims with evidence (v0.12.3+)
│       └── scored_claims.jsonl        # Confidence-scored claims (v0.12.5+)
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

### MinerU conversion

High-quality scientific PDF parser with excellent table recognition (TEDS 93.42%):

```bash
pip install 'research-pipeline[mineru]'
# or
pipx inject research-pipeline magic-pdf
```

MinerU supports three parse modes: `auto` (recommended), `ocr` (scanned docs),
and `txt` (text extraction only). Configure in `config.toml`:

```toml
[conversion]
backend = "mineru"

[conversion.mineru]
parse_method = "auto"
timeout_seconds = 600
```

### Google Scholar access

The free `scholarly` library may be blocked by Google. For reliable access,
use SerpAPI:

```bash
uv sync --extra serpapi
```

Set your API key in the config or environment.

## Multi-account support

Each online conversion backend supports **multiple accounts**. When a quota or
rate limit is hit on one account, the pipeline automatically rotates to the next
account before falling through to the next backend service.

### Configuring multiple accounts

Use TOML array-of-tables (`[[...]]`) syntax to define multiple accounts for
any online backend:

```toml
[conversion]
backend = "mathpix"

# Multiple Mathpix accounts — tried in order
[[conversion.mathpix.accounts]]
app_id = "account-1-id"
app_key = "account-1-key"

[[conversion.mathpix.accounts]]
app_id = "account-2-id"
app_key = "account-2-key"

# Multiple Datalab accounts
[[conversion.datalab.accounts]]
api_key = "datalab-key-1"
mode = "fast"

[[conversion.datalab.accounts]]
api_key = "datalab-key-2"
mode = "accurate"

# LlamaParse
[[conversion.llamaparse.accounts]]
api_key = "llama-key-1"
tier = "agentic"                # per-account tier override

[[conversion.llamaparse.accounts]]
api_key = "llama-key-2"
tier = "cost-effective"

# Mistral OCR
[[conversion.mistral_ocr.accounts]]
api_key = "mistral-key-1"

[[conversion.mistral_ocr.accounts]]
api_key = "mistral-key-2"
model = "mistral-ocr-latest"

# OpenAI Vision
[[conversion.openai_vision.accounts]]
api_key = "openai-key-1"

[[conversion.openai_vision.accounts]]
api_key = "openai-key-2"
model = "gpt-4o-mini"
```

**Backward compatibility**: If no `[[...accounts]]` are defined, the top-level
single credentials (e.g., `[conversion.mathpix] app_id = "..."`) are used as
before.

## Fallback backends

Configure automatic cross-service failover with `fallback_backends`. When
the primary backend (and all its accounts) is exhausted, the pipeline
tries the next service:

```toml
[conversion]
backend = "mathpix"
fallback_backends = ["datalab", "mistral_ocr", "openai_vision"]
```

The execution order for the example above:

1. Mathpix account 1 → Mathpix account 2 → ...
2. Datalab account 1 → Datalab account 2 → ...
3. Mistral OCR account 1 → Mistral OCR account 2 → ...
4. OpenAI Vision account 1 → OpenAI Vision account 2 → ...

The pipeline detects quota/rate-limit errors automatically (HTTP 429, "rate
limit", "quota exceeded", "insufficient credits", etc.) and logs each rotation.
Non-quota errors also trigger fallback to the next backend.
