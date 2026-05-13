# User Manual: research-pipeline

> **Version:** 0.17.14 · [GitHub](https://github.com/grammy-jiang/research-pipeline) · [Documentation](https://grammy-jiang.github.io/research-pipeline/)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
   - [2.1 Basic install](#21-basic-install)
   - [2.2 Optional extras](#22-optional-extras)
   - [2.3 Verify installation](#23-verify-installation)
3. [Configuration](#3-configuration)
   - [3.1 Create config.toml](#31-create-configtoml)
   - [3.2 Environment variable](#32-environment-variable)
   - [3.3 Key configuration options](#33-key-configuration-options)
   - [3.4 API key setup](#34-api-key-setup)
4. [Quick Start](#4-quick-start)
   - [4.1 Your first research run](#41-your-first-research-run)
   - [4.2 Reading your results](#42-reading-your-results)
5. [Academic Paper Research Pipeline](#5-academic-paper-research-pipeline)
   - [5.1 Running end-to-end](#51-running-end-to-end)
   - [5.2 Choosing a pipeline profile](#52-choosing-a-pipeline-profile)
   - [5.3 Running individual stages](#53-running-individual-stages)
   - [5.4 Resuming a failed run](#54-resuming-a-failed-run)
   - [5.5 Working with multiple sources](#55-working-with-multiple-sources)
   - [5.6 PDF conversion options](#56-pdf-conversion-options)
   - [5.7 Exporting results](#57-exporting-results)
   - [5.8 Iterative deep research](#58-iterative-deep-research)
6. [Daily AI Intelligence Briefing](#6-daily-ai-intelligence-briefing)
   - [6.1 Overview of the briefing pipeline](#61-overview-of-the-briefing-pipeline)
   - [6.2 Configuring briefing sources](#62-configuring-briefing-sources)
   - [6.3 Running the daily brief](#63-running-the-daily-brief)
   - [6.4 Understanding the output](#64-understanding-the-output)
   - [6.5 Hot-topic dossiers](#65-hot-topic-dossiers)
   - [6.6 Weekly synthesis](#66-weekly-synthesis)
   - [6.7 Exporting to Obsidian](#67-exporting-to-obsidian)
   - [6.8 Feedback and personalization](#68-feedback-and-personalization)
7. [Using with AI Assistants (MCP + Skills)](#7-using-with-ai-assistants-mcp--skills)
   - [7.1 MCP server setup](#71-mcp-server-setup)
   - [7.2 Using research-workflow via MCP](#72-using-research-workflow-via-mcp)
   - [7.3 Installing the AI skill](#73-installing-the-ai-skill)
   - [7.4 Sub-agent workflows](#74-sub-agent-workflows)
8. [Advanced Features](#8-advanced-features)
   - [8.1 Citation graph expansion](#81-citation-graph-expansion)
   - [8.2 Quality scoring](#82-quality-scoring)
   - [8.3 Semantic re-ranking](#83-semantic-re-ranking)
   - [8.4 Feedback loop](#84-feedback-loop)
   - [8.5 Knowledge graph](#85-knowledge-graph)
   - [8.6 Watching for new papers](#86-watching-for-new-papers)
9. [Troubleshooting](#9-troubleshooting)
   - [9.1 No results from search](#91-no-results-from-search)
   - [9.2 PDF download failures](#92-pdf-download-failures)
   - [9.3 Conversion errors](#93-conversion-errors)
   - [9.4 MCP connection issues](#94-mcp-connection-issues)
   - [9.5 Rate limiting](#95-rate-limiting)
   - [9.6 Common error messages and fixes](#96-common-error-messages-and-fixes)
10. [Command Reference Summary](#10-command-reference-summary)

---

## 1. Introduction

**research-pipeline** is a command-line tool for two distinct workflows:

1. **Academic paper research** — given a research topic, it searches arXiv and
   other academic databases, screens for relevance, downloads PDFs, converts them
   to Markdown, and synthesises a structured literature review with evidence citations.

2. **Daily AI intelligence briefing** — polls configurable sources (RSS feeds,
   GitHub releases, arXiv events, HackerNews, and more), deduplicates and ranks
   events by technical significance, and generates a concise daily digest — with
   optional export to Obsidian.

**When to use it:**

| Goal | Use |
|------|-----|
| Literature review on an AI topic | `research-pipeline run "topic"` |
| Staying current with AI developments | `research-pipeline brief run` |
| Standalone PDF → Markdown | `research-pipeline convert-file paper.pdf` |
| AI assistant with academic search tools | `research-pipeline mcp serve` |

> **Note:** Both pipelines are offline-first. The academic pipeline only contacts
> arXiv and academic APIs; the briefing pipeline only contacts explicitly
> configured sources. No telemetry or data is sent to third parties.

---

## 2. Installation

### 2.1 Basic install

```bash
# Using pip
pip install research-pipeline

# Using uv (recommended)
uv add research-pipeline
```

**Requirements:** Python 3.12 or later.

### 2.2 Optional extras

Install extras for additional capabilities:

| Extra | Purpose | License |
|-------|---------|---------|
| `docling` | High-quality PDF → Markdown conversion (recommended) | MIT |
| `marker` | Highest-accuracy PDF → Markdown conversion | GPL-3.0 |
| `pymupdf4llm` | Fast CPU-only PDF → Markdown conversion | AGPL |
| `mineru` | Scientific PDF parser (TEDS 93.42% table accuracy) | MIT |
| `mathpix` | Mathpix cloud OCR — best LaTeX (1 000 free pages/month) | Proprietary |
| `datalab` | Datalab hosted Marker (\$5 free credit) | Proprietary |
| `llamaparse` | LlamaParse cloud parsing (1 000 free pages/day) | Proprietary |
| `mistral-ocr` | Mistral Document AI OCR (per-token, free credits available) | Proprietary |
| `openai-vision` | OpenAI GPT-4o vision (per-token) | Proprietary |
| `scholar` | Google Scholar search via the `scholarly` library | MIT |
| `serpapi` | Google Scholar via SerpAPI (requires API key) | MIT |
| `reranker` | SPECTER2 semantic re-ranking for higher-precision screening | MIT |
| `llm` | LLM integration for summarisation and claim analysis | — |

**Install multiple extras at once:**

```bash
# Recommended full-featured install
pip install "research-pipeline[docling,scholar,reranker]"

# Development install from source
git clone https://github.com/grammy-jiang/research-pipeline.git
cd research-pipeline
uv sync --extra dev --extra docling --extra scholar
```

> **Tip:** Start with `docling` for PDF conversion. It requires no API keys and
> gives good results on most scientific papers. Add `reranker` to improve
> screening precision when working on niche or technical topics.

### 2.3 Verify installation

```bash
research-pipeline --version
# Expected: research-pipeline 0.17.14

research-pipeline --help
```

---

## 3. Configuration

### 3.1 Create config.toml

Most features work with defaults, but you need a config file to set API keys
and tune behaviour.

```bash
# Copy the annotated example
cp config.example.toml config.toml
# Or download it directly
curl -o config.toml https://raw.githubusercontent.com/grammy-jiang/research-pipeline/main/config.example.toml
```

The tool looks for `config.toml` in the **current working directory** by
default. Pass `--config /path/to/config.toml` on any command to override.

### 3.2 Environment variable

Set `RESEARCH_PIPELINE_CONFIG` to point to your config file from any directory:

```bash
export RESEARCH_PIPELINE_CONFIG=/home/user/.config/research-pipeline/config.toml
research-pipeline run "transformer attention"
```

You can also override individual secrets via environment variables without
putting them in the config file (see [§3.4](#34-api-key-setup)).

### 3.3 Key configuration options

**`[pipeline]` — top-level pipeline behaviour**

| Key | Default | Description |
|-----|---------|-------------|
| `profile` | `"standard"` | Pipeline profile: `quick`, `standard`, `deep`, `auto` |
| `ter_max_iterations` | `3` | Max THINK→EXECUTE→REFLECT iterations for iterative research |
| `memory_working_capacity` | `50` | Max items in working memory per stage |

**`[search]` — search parameters**

| Key | Default | Description |
|-----|---------|-------------|
| `primary_months` | `6` | Months of recency for primary search window |
| `fallback_months` | `12` | Extended window when primary yields too few results |
| `max_query_variants` | `5` | Number of query variants to generate and search |
| `min_candidates` | `40` | Minimum candidates before proceeding to screening |

**`[sources]` — multi-source search**

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `["arxiv"]` | Active sources: `arxiv`, `semantic_scholar`, `openalex`, `dblp`, `scholar`, `huggingface` |
| `scholar_backend` | `"scholarly"` | `scholarly` (free) or `serpapi` (paid, faster) |
| `semantic_scholar_api_key` | `""` | Optional — raises rate limits substantially |
| `openalex_api_key` | `""` | Optional polite-pool key for OpenAlex |

**`[screen]` — relevance screening**

| Key | Default | Description |
|-----|---------|-------------|
| `cheap_top_k` | `50` | Candidates kept after BM25 scoring |
| `download_top_n` | `8` | Papers selected for download |
| `final_score_threshold` | `0.70` | Minimum BM25 score to include |
| `use_semantic_reranking` | `false` | Enable SPECTER2 re-ranking (requires `reranker` extra) |
| `diversity` | `false` | Enable MMR diversity-aware shortlisting |

**`[download]` — PDF downloading**

| Key | Default | Description |
|-----|---------|-------------|
| `max_per_run` | `20` | Hard cap on PDFs downloaded per run |

**`[conversion]` — PDF → Markdown**

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `"docling"` | Primary backend: `docling`, `marker`, `pymupdf4llm`, `mineru`, `mathpix`, `datalab`, `llamaparse`, `mistral_ocr`, `openai_vision` |
| `fallback_backends` | `[]` | Ordered list of backends to try when primary fails |
| `timeout_seconds` | `300` | Per-file conversion timeout |

**`[llm]` — LLM integration** (optional)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable LLM-powered summarisation |
| `provider` | `"ollama"` | `ollama` (local) or `openai` (API) |
| `model` | `""` | Model name (empty = provider default) |
| `api_key` | `""` | API key for OpenAI-compatible providers |

**`[quality]` — paper quality scoring**

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable citation/venue/author quality scoring |
| `citation_weight` | `0.35` | Weight for citation impact |
| `venue_weight` | `0.25` | Weight for venue reputation (CORE ranking) |
| `author_weight` | `0.25` | Weight for author h-index |
| `recency_weight` | `0.15` | Weight for publication recency |

**`[incremental]` — cross-run deduplication**

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Skip papers seen in previous runs |
| `global_index_path` | `""` | SQLite index path (default: `~/.cache/research-pipeline/paper_index.db`) |
| `reuse_artifacts` | `true` | Symlink existing PDFs instead of re-downloading |

### 3.4 API key setup

You can store API keys in `config.toml` or as environment variables. The
environment variable names follow the pattern `RESEARCH_PIPELINE_<SECTION>_<KEY>` for
most keys, but some are named explicitly in the example config.

| Service | Config key | Environment variable |
|---------|-----------|---------------------|
| Semantic Scholar | `sources.semantic_scholar_api_key` | `RESEARCH_PIPELINE_SEMANTIC_SCHOLAR_API_KEY` |
| SerpAPI (Scholar) | `sources.serpapi_key` | `RESEARCH_PIPELINE_SERPAPI_KEY` |
| OpenAlex | `sources.openalex_api_key` | `RESEARCH_PIPELINE_OPENALEX_API_KEY` |
| Datalab | `conversion.datalab.api_key` | `RESEARCH_PIPELINE_DATALAB_API_KEY` |
| LlamaParse | `conversion.llamaparse.api_key` | `RESEARCH_PIPELINE_LLAMAPARSE_API_KEY` |
| Mathpix | `conversion.mathpix.app_id` / `.app_key` | `RESEARCH_PIPELINE_MATHPIX_APP_ID` / `RESEARCH_PIPELINE_MATHPIX_APP_KEY` |
| Mistral OCR | `conversion.mistral_ocr.api_key` | `RESEARCH_PIPELINE_MISTRAL_API_KEY` |
| OpenAI vision | `conversion.openai_vision.api_key` | `RESEARCH_PIPELINE_OPENAI_API_KEY` |

> **Note:** `config.toml` is gitignored by default in the project template.
> Never commit API keys to source control.

---

## 4. Quick Start

### 4.1 Your first research run

Five minutes to your first literature review:

```bash
# 1. Install with recommended extras
pip install "research-pipeline[docling]"

# 2. (Optional) Create a config file
cp config.example.toml config.toml   # then edit as needed

# 3. Run the pipeline
research-pipeline run "attention mechanisms in transformer models"
```

The pipeline will:
1. Normalise your topic into structured search queries
2. Search arXiv (and any other enabled sources)
3. Score and shortlist the most relevant papers
4. Download PDFs for the top candidates
5. Convert PDFs to Markdown
6. Extract key sections
7. Generate per-paper summaries and a cross-paper synthesis report

Progress is printed to the terminal. A typical run takes 2–5 minutes.

**Find your results:**

```bash
# List all runs
research-pipeline inspect

# Show details for a specific run
research-pipeline inspect --run-id <RUN_ID>
```

Run output is stored under `./runs/<RUN_ID>/`.

### 4.2 Reading your results

Each run produces these key artifacts:

| Path | Contents |
|------|----------|
| `runs/<ID>/query_plan.json` | Structured search queries generated from your topic |
| `runs/<ID>/candidates.jsonl` | All papers found across sources |
| `runs/<ID>/screened.jsonl` | Shortlisted papers with relevance scores |
| `runs/<ID>/convert/<ID>.md` | Converted paper text (one file per paper) |
| `runs/<ID>/synthesis.md` | **Main output:** cross-paper synthesis report |
| `runs/<ID>/summaries/<ID>.md` | Per-paper summary |
| `runs/<ID>/run_manifest.json` | Stage completion status and artifact hashes |

Open `synthesis.md` (or `synthesis.html` after export) to read your literature review.

```bash
# Export to HTML for easy reading in a browser
research-pipeline export-html --run-id <RUN_ID>
# Opens as runs/<ID>/synthesis.html

# Export to BibTeX for LaTeX use
research-pipeline export-bibtex --run-id <RUN_ID>
# Writes runs/<ID>/screened.bib
```

---

## 5. Academic Paper Research Pipeline

### 5.1 Running end-to-end

The `run` command orchestrates all seven stages in sequence:

```bash
# Basic run with default settings (standard profile, arXiv only)
research-pipeline run "LLM reasoning and chain-of-thought"

# Run with debug output
research-pipeline run "diffusion models" --verbose

# Use a custom config
research-pipeline run "RAG architectures" --config /path/to/config.toml

# Save to a custom workspace directory
research-pipeline run "embodied AI" --workspace /data/runs

# Pin to a specific run ID (reproducible re-runs)
research-pipeline run "graph neural networks" --run-id my-gnn-run-01
```

### 5.2 Choosing a pipeline profile

The `--profile` flag controls which stages execute and how deep the analysis goes.

| Profile | Stages | Use when |
|---------|--------|----------|
| `quick` | plan → search → screen → summarise | You need fast results from abstracts only (no PDFs) |
| `standard` | Full 7-stage pipeline | Default; balanced quality and speed |
| `deep` | Full pipeline + expand + quality + claim analysis + TER loop | Thorough literature review with iterative gap-filling |
| `auto` | Detected from query complexity | Let the pipeline decide |

```bash
# Quick abstract-only survey (no downloads)
research-pipeline run "vector database benchmarks" --profile quick

# Standard full pipeline (default)
research-pipeline run "multimodal LLMs" --profile standard

# Deep iterative research with up to 3 refinement rounds
research-pipeline run "AI alignment methods" --profile deep

# Let the pipeline decide based on query complexity
research-pipeline run "neural architecture search" --profile auto
```

> **Tip:** Use `quick` for an initial survey to understand the space, then
> re-run with `standard` or `deep` once you know your focus area.

### 5.3 Running individual stages

Run stages one at a time when you need fine-grained control:

```bash
# Stage 1: Create a query plan from your topic
research-pipeline plan "low-rank adaptation of large language models"
# Outputs: runs/<ID>/query_plan.json
# Prints the run ID — use it in subsequent commands

# Stage 2: Search all configured sources
research-pipeline search --run-id <RUN_ID>

# Stage 2 (alt): Search a specific source
research-pipeline search --run-id <RUN_ID> --source semantic_scholar

# Stage 3: Screen and shortlist candidates
research-pipeline screen --run-id <RUN_ID>

# Stage 3 (with diversity): balance relevance and topic diversity
research-pipeline screen --run-id <RUN_ID> --diversity --diversity-lambda 0.5

# Stage 4: Download PDFs
research-pipeline download --run-id <RUN_ID>

# Stage 5: Convert PDFs to Markdown
research-pipeline convert --run-id <RUN_ID>

# Stage 5 (with backend override)
research-pipeline convert --run-id <RUN_ID> --backend marker

# Stage 6: Extract sections and chunks
research-pipeline extract --run-id <RUN_ID>

# Stage 7: Generate summaries and synthesis
research-pipeline summarize --run-id <RUN_ID>
```

> **Tip:** After running `plan`, copy the printed run ID and export it:
> `export RUN_ID=<value>`. Then use `--run-id $RUN_ID` for all subsequent stages.

### 5.4 Resuming a failed run

If a stage fails partway through, resume from where it stopped:

```bash
# Resume the full pipeline from where it stopped
research-pipeline run "topic" --run-id <EXISTING_RUN_ID> --resume

# Resume a specific stage
research-pipeline search --run-id <RUN_ID> --resume
research-pipeline screen --run-id <RUN_ID> --resume

# Retry only failed downloads (without re-downloading successful ones)
research-pipeline download --run-id <RUN_ID> --retry-failed

# Re-convert forcing overwrite of existing Markdown files
research-pipeline convert --run-id <RUN_ID> --force
```

Check the current state of a run at any time:

```bash
research-pipeline inspect --run-id <RUN_ID>
```

### 5.5 Working with multiple sources

By default, only arXiv is searched. Enable additional sources in `config.toml`:

```toml
[sources]
enabled = ["arxiv", "semantic_scholar", "openalex", "dblp"]
semantic_scholar_api_key = "your-key-here"   # optional but recommended  # pragma: allowlist secret
```

Or override at run time:

```bash
# Search all configured sources
research-pipeline run "contrastive learning" --source all

# Search specific sources
research-pipeline search --run-id <RUN_ID> --source semantic_scholar
research-pipeline search --run-id <RUN_ID> --source "arxiv,openalex"
```

Available sources: `arxiv`, `semantic_scholar`, `openalex`, `dblp`,
`scholar` (Google Scholar), `huggingface` (HuggingFace daily papers).

> **Note:** Google Scholar (`scholar`) requires either the `scholarly` Python
> package (free, slow) or a SerpAPI key (paid, faster). DBLP and OpenAlex
> are free and do not require API keys.

**Enrich candidates with missing abstracts:**

```bash
research-pipeline enrich --run-id <RUN_ID>
```

This queries Semantic Scholar to fill in any missing abstracts before screening.

### 5.6 PDF conversion options

Three local backends are available:

| Backend | Speed | Quality | Requires |
|---------|-------|---------|----------|
| `pymupdf4llm` | Fastest | Good | `pip install research-pipeline[pymupdf4llm]` |
| `docling` | Medium | Very good | `pip install research-pipeline[docling]` |
| `marker` | Slowest | Best (GPL) | `pip install research-pipeline[marker]` |

Five cloud backends are also available for cases where local conversion fails
or higher quality is needed: `mathpix`, `datalab`, `llamaparse`,
`mistral_ocr`, `openai_vision`. Each requires an API key
(see [§3.4](#34-api-key-setup)).

**Two-tier conversion workflow** (recommended for large runs):

```bash
# Step 1: Fast rough conversion for all papers (CPU only, no extra install needed)
research-pipeline convert-rough --run-id <RUN_ID>

# Step 2: High-quality conversion for specific papers that need it
research-pipeline convert-fine --run-id <RUN_ID> --paper-ids 2401.12345,2401.67890
```

**Standalone PDF conversion** (no pipeline workspace needed):

```bash
research-pipeline convert-file paper.pdf
research-pipeline convert-file paper.pdf -o ./output/ --backend marker
```

**Configure fallback backends** in `config.toml`:

```toml
[conversion]
backend = "docling"
fallback_backends = ["pymupdf4llm"]   # try pymupdf4llm if docling fails
```

### 5.7 Exporting results

**HTML** — self-contained, browser-readable report:

```bash
research-pipeline export-html --run-id <RUN_ID>
# Output: runs/<ID>/synthesis.html

# Convert any Markdown report to HTML
research-pipeline export-html --markdown my-report.md -o report.html --title "My Review"
```

**BibTeX** — for use in LaTeX documents:

```bash
research-pipeline export-bibtex --run-id <RUN_ID>
# Output: runs/<ID>/screened.bib

# Export from a different stage (more papers)
research-pipeline export-bibtex --run-id <RUN_ID> --stage search -o all-papers.bib
```

**Structured report templates:**

```bash
# Survey (default)
research-pipeline report --run-id <RUN_ID>

# Gap analysis
research-pipeline report --run-id <RUN_ID> --template gap_analysis

# Literature review
research-pipeline report --run-id <RUN_ID> --template lit_review

# Executive summary
research-pipeline report --run-id <RUN_ID> --template executive

# Custom Jinja2 template
research-pipeline report --run-id <RUN_ID> --custom-template my_template.j2
```

**JSON output from summarize:**

```bash
# Output synthesis as JSON
research-pipeline summarize --run-id <RUN_ID> --output-format json

# Output as BibTeX directly
research-pipeline summarize --run-id <RUN_ID> --output-format bibtex
```

**Evidence aggregation** (strips rhetorical hedging for clean bullet points):

```bash
research-pipeline aggregate --run-id <RUN_ID>
research-pipeline aggregate --run-id <RUN_ID> --min-pointers 1 --format json
```

**Validate report completeness:**

```bash
research-pipeline validate --run-id <RUN_ID>
research-pipeline validate --report report.md
```

This checks for the 14 required sections, confidence annotations, evidence
citations, and gap classifications.

### 5.8 Iterative deep research

Use `--profile deep` or `--system-building` to enable the
THINK→EXECUTE→REFLECT (TER) iterative loop. The pipeline automatically
identifies gaps in the synthesis and runs additional searches to fill them.

```bash
# Deep profile with default 3 iterations
research-pipeline run "AI agent memory systems" --profile deep

# Control the number of TER iterations
research-pipeline run "LLM safety techniques" --profile deep --ter-iterations 5

# Interactive mode: pause at each gate for human review
research-pipeline run "neural architecture search" --profile deep --interactive
```

After a deep run, **compare two iterations** to see what changed:

```bash
research-pipeline compare --run-a <RUN_ID_1> --run-b <RUN_ID_2>
```

**Evaluate multi-run coherence** (detect contradictions across runs):

```bash
research-pipeline coherence <RUN_ID_1> <RUN_ID_2> <RUN_ID_3>
```

---

## 6. Daily AI Intelligence Briefing

### 6.1 Overview of the briefing pipeline

The `brief` sub-command runs a separate four-stage pipeline:

```
poll → rank → generate-daily → validate
```

| Stage | What it does |
|-------|-------------|
| `poll` | Fetches events from configured sources (RSS, GitHub, arXiv, HN…) |
| `rank` | Deduplicates, clusters, and ranks events by technical significance |
| `generate-daily` | Renders a structured Markdown digest from ranked clusters |
| `validate` | Checks the digest meets format and content requirements |

Run all four stages at once:

```bash
research-pipeline brief run
```

### 6.2 Configuring briefing sources

Create a source registry file (JSON or TOML) that lists your intelligence sources.
You can maintain this separately from the main `config.toml`.

**Example `briefing-sources.toml`:**

```toml
[[sources]]
source_id = "arxiv_cs_ai"
source_name = "arXiv CS.AI daily"
source_class = "academic_source"
access_method = "rss_atom"
url = "https://rss.arxiv.org/rss/cs.AI"
enabled = true
cadence = "daily"

[[sources]]
source_id = "github_huggingface"
source_name = "HuggingFace releases"
source_class = "implementation_source"
access_method = "github_releases"
url = "https://github.com/huggingface/transformers"
enabled = true
cadence = "daily"

[[sources]]
source_id = "hacker_news"
source_name = "HackerNews AI"
source_class = "technical_discussion"
access_method = "rss_atom"
url = "https://hnrss.org/newest?q=AI+LLM"
enabled = true
cadence = "daily"
```

Source classes: `primary_artifact`, `academic_source`, `implementation_source`,
`technical_discussion`, `social_signal`, `media_news`, `newsletter`, `video_audio`.

### 6.3 Running the daily brief

```bash
# Run the full briefing pipeline (poll → rank → generate → validate)
research-pipeline brief run

# Run with a custom source registry
research-pipeline brief run --registry briefing-sources.toml

# Run for a specific date (ISO format)
research-pipeline brief run --date 2025-07-04

# Use a custom workspace directory
research-pipeline brief run --workspace ~/briefings

# Run individual stages
research-pipeline brief poll --registry briefing-sources.toml
research-pipeline brief rank
research-pipeline brief generate-daily
research-pipeline brief validate
```

Results are written to `./workspace/briefings/<YYYY-MM-DD>/`.

**Resume from a failed stage:**

```bash
research-pipeline brief resume --date 2025-07-04 --from-stage generate-daily
```

### 6.4 Understanding the output

A briefing run produces these artifacts under `workspace/briefings/<DATE>/`:

| Path | Contents |
|------|----------|
| `raw/` | Raw polled events per source |
| `normalized/` | Normalised event records |
| `ranked_clusters.jsonl` | Ranked and clustered topic groups |
| `reports/daily.md` | **The daily brief** — your main output |
| `reports/validation.json` | Validation results |
| `memory/topics.db` | Topic memory for personalisation |
| `feedback/feedback.db` | Your feedback history |

The daily brief is a structured Markdown document with:
- Top-ranked clusters by technical significance
- Source attribution and canonical URLs
- Signal metadata (novelty, recency, source class)

### 6.5 Hot-topic dossiers

When a cluster is particularly important, generate an in-depth dossier:

```bash
# Generate a dossier for a specific cluster
research-pipeline brief dossier --cluster <CLUSTER_ID>

# Auto-select the top candidate cluster
research-pipeline brief dossier --auto

# Auto-select top 3 candidates
research-pipeline brief dossier --auto --max-count 3
```

Dossiers are written to `workspace/briefings/<DATE>/reports/dossiers/`.

### 6.6 Weekly synthesis

Generate a trend memo from the past week's daily briefs:

```bash
research-pipeline brief weekly-synthesis --week 2025-W27

# Save to a custom path
research-pipeline brief weekly-synthesis --week 2025-W27 -o weekly/2025-W27.md
```

The weekly memo is saved to `workspace/briefings/weekly/<WEEK>.md`.

### 6.7 Exporting to Obsidian

Export your daily brief as linked Obsidian notes (daily note + per-topic notes
+ per-source notes):

```bash
research-pipeline brief export-obsidian --vault ~/my-vault

# Dry-run: see what would be created without writing
research-pipeline brief export-obsidian --vault ~/my-vault --dry-run

# Export for a specific date
research-pipeline brief export-obsidian --vault ~/my-vault --date 2025-07-04
```

Notes are created in your vault under:
- `Briefings/Daily/<DATE>.md` — daily note
- `Briefings/Topics/<TOPIC>.md` — per-topic notes
- `Briefings/Sources/<SOURCE>.md` — per-source notes

### 6.8 Feedback and personalization

Record feedback to improve future rankings. Feedback accumulates and adjusts
cluster/topic/source weights via an ELO-style learning algorithm.

**Available feedback signals:**

| Signal | Effect |
|--------|--------|
| `keep` | Preserve this item in future briefs |
| `hide` | Suppress this item |
| `more_like_this` | Boost similar content |
| `less_like_this` | Reduce similar content |
| `too_noisy` | This source is too noisy |
| `already_known` | Skip — I already knew this |
| `not_actionable` | Interesting but nothing to act on |
| `useful` | Positive general signal |
| `neutral` | No preference |
| `not_useful` | Negative general signal |

```bash
# Boost a source
research-pipeline brief feedback --signal more_like_this --source arxiv_cs_ai

# Suppress a topic
research-pipeline brief feedback --signal less_like_this --topic "crypto_nft"

# Hide a specific cluster with a reason
research-pipeline brief feedback --signal hide --cluster <CLUSTER_ID> \
    --reason "off-topic for my interests"

# Show all recorded feedback
research-pipeline brief feedback --signal keep --show

# List conflicting feedback entries
research-pipeline brief feedback --signal keep --conflicts
```

**Apply preference adjustments:**

```bash
# Compute and apply preference weights from accumulated feedback
research-pipeline brief preferences

# Rollback a specific adjustment
research-pipeline brief preferences --rollback <ADJUSTMENT_ID>
```

**Review topic alias suggestions:**

The pipeline may suggest that two topic IDs refer to the same concept.
Review these periodically:

```bash
# List pending alias suggestions
research-pipeline brief topic-aliases

# Approve a suggestion
research-pipeline brief topic-aliases --approve <SUGGESTION_ID>

# Reject a suggestion
research-pipeline brief topic-aliases --reject <SUGGESTION_ID> --review "these are different"
```

**Compare source registries** to evaluate adding a new source:

```bash
research-pipeline brief compare-sources \
    --base-registry current.toml \
    --expanded-registry with-new-source.toml \
    --date 2025-07-04
```

---

## 7. Using with AI Assistants (MCP + Skills)

### 7.1 MCP server setup

The MCP (Model Context Protocol) server exposes all pipeline capabilities as
tools that AI assistants can invoke directly in conversations.

**Start the server:**

```bash
# Start MCP server on stdio (the standard transport)
research-pipeline mcp serve

# Print a config snippet for your MCP client
research-pipeline mcp config
```

**Configure your MCP client.** For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-pipeline": {
      "command": "research-pipeline",
      "args": ["mcp", "serve"]
    }
  }
}
```

For path-based installs:

```json
{
  "mcpServers": {
    "research-pipeline": {
      "command": "/path/to/venv/bin/research-pipeline",
      "args": ["mcp", "serve"]
    }
  }
}
```

The server provides **51 tools**, **15 resources** (run artifact URI templates),
and **6 prompt templates** for common research workflows.

### 7.2 Using research-workflow via MCP

The `research_workflow` tool is the recommended way to run the full pipeline
from an AI assistant. It includes 6 harness layers:

1. **Telemetry** — three-channel logging (cognitive/operational/contextual)
2. **Context engineering** — token budget management and paper compaction
3. **Governance** — schema-level state machine with verify-before-commit gates
4. **Verification** — structural output validation at each stage
5. **Monitoring** — doom-loop detection and iteration drift tracking
6. **Recovery** — automatic crash recovery via persistent state

To trigger it from an AI conversation:

```
Run a research pipeline on "memory architectures for AI agents"
```

The assistant will use the `research_workflow` MCP tool with structured
elicitation at decision gates.

### 7.3 Installing the AI skill

The skill installs a `SKILL.md` file that teaches Claude Code, GitHub Copilot,
and Codex how to use this tool effectively.

```bash
# Install skill + sub-agents + MCP config snippet
research-pipeline setup

# Install with symlinks (useful during development — changes are reflected immediately)
research-pipeline setup --symlink

# Force overwrite existing files
research-pipeline setup --force

# Install skill only (skip sub-agents)
research-pipeline setup --skip-agents

# Install sub-agents only (skip skill)
research-pipeline setup --skip-skill
```

This installs:
- Skill → `~/.claude/skills/research-pipeline/` and `~/.codex/skills/research-pipeline/`
- Sub-agents → `~/.claude/agents/`
- MCP config snippet → `~/.config/research-pipeline/mcp.json`

### 7.4 Sub-agent workflows

Three specialist sub-agents are installed by `research-pipeline setup`:

| Agent | Role |
|-------|------|
| `paper-analyzer` | Deep analysis of a single paper's claims and methods |
| `paper-screener` | Relevance screening against a research query |
| `paper-synthesizer` | Cross-paper synthesis and gap identification |

These agents are invoked automatically by the skill when running deep research
from Claude Code or GitHub Copilot.

**Prepare analysis tasks for the paper-analyzer agent:**

```bash
# Prepare task prompts for all converted papers
research-pipeline analyze --run-id <RUN_ID>

# Prepare for specific papers only
research-pipeline analyze --run-id <RUN_ID> --paper-ids 2401.12345,2401.67890

# Validate collected analysis results after the agent has processed them
research-pipeline analyze --run-id <RUN_ID> --collect
```

---

## 8. Advanced Features

### 8.1 Citation graph expansion

Expand your corpus by following citation links from seed papers:

```bash
# Single-hop expansion (direct citations and references)
research-pipeline expand \
    --run-id <RUN_ID> \
    --paper-ids 2401.12345,2401.67890 \
    --direction both

# Citations only
research-pipeline expand --run-id <RUN_ID> --paper-ids 2401.12345 --direction citations

# BFS multi-hop expansion (2 hops, BM25-pruned)
research-pipeline expand \
    --run-id <RUN_ID> \
    --paper-ids 2401.12345 \
    --bfs-depth 2 \
    --bfs-query "transformer,attention,self-attention"

# Snowball sampling: iterative bidirectional expansion with budget-aware stopping
research-pipeline expand \
    --run-id <RUN_ID> \
    --paper-ids 2401.12345 \
    --snowball \
    --snowball-max-papers 200 \
    --bfs-query "LoRA,adapter,fine-tuning"
```

Expansion modes:

| Mode | Command | When to use |
|------|---------|------------|
| Single-hop | default | Quick survey of direct citations |
| BFS | `--bfs-depth 2` | Systematic multi-hop exploration |
| Snowball | `--snowball` | Comprehensive corpus building with stopping criteria |

### 8.2 Quality scoring

Score papers on citation impact, venue reputation, author credibility, and recency.
Enable in `config.toml` (`quality.enabled = true`) or run standalone:

```bash
research-pipeline quality --run-id <RUN_ID>
```

Results are added to each candidate's metadata. View via:

```bash
research-pipeline inspect --run-id <RUN_ID>
```

**Deep claim analysis:**

```bash
# Decompose summaries into atomic claims with evidence classification
research-pipeline analyze-claims --run-id <RUN_ID>

# Score confidence for each claim
research-pipeline score-claims --run-id <RUN_ID>
```

**Recall/Reasoning/Presentation diagnostic:**

```bash
research-pipeline rrp --report synthesis.md --shortlist shortlist.json
```

This diagnoses the bottleneck in your synthesis quality along three axes:
information recall, reasoning quality, and presentation clarity.

### 8.3 Semantic re-ranking

SPECTER2-based semantic similarity scoring improves screening precision, especially
for niche or highly technical topics.

Enable in `config.toml`:

```toml
[screen]
use_semantic_reranking = true
embedding_model = "allenai/specter2"
```

Requires: `pip install research-pipeline[reranker]`

The first run will download the SPECTER2 model (~440 MB). Subsequent runs use the
cached model.

### 8.4 Feedback loop

Record paper-level accept/reject decisions to improve future runs. The BM25
screening weights are adjusted via ELO-style learning.

```bash
# Accept papers that were relevant
research-pipeline feedback \
    --run-id <RUN_ID> \
    --accept 2401.12345 \
    --accept 2401.67890

# Reject an off-topic paper with a reason
research-pipeline feedback \
    --run-id <RUN_ID> \
    --reject 2401.99999 \
    --reason "survey paper, not primary research"

# Show feedback statistics
research-pipeline feedback --run-id <RUN_ID> --show

# Recompute adjusted BM25 weights from all feedback
research-pipeline feedback --run-id <RUN_ID> --adjust
```

### 8.5 Knowledge graph

Build a persistent knowledge graph from pipeline results for structured querying
across runs.

```bash
# Ingest a run's results into the KG
research-pipeline kg-ingest --run-id <RUN_ID>

# Show graph statistics (entity and triple counts)
research-pipeline kg-stats

# Query an entity and its relations
research-pipeline kg-query 2401.12345

# Memory tier summary (working + episodic + semantic)
research-pipeline memory-stats

# List recent pipeline runs in episodic memory
research-pipeline memory-episodes --limit 10

# Search episodic memory for a past topic
research-pipeline memory-search "transformer attention"

# Consolidate episodic memory: promote recurring findings to rules
research-pipeline consolidate

# Dry-run to see what would be consolidated
research-pipeline consolidate --dry-run
```

### 8.6 Watching for new papers

Monitor arXiv for new papers matching your interests:

```bash
# Check for new papers (looks back 7 days by default)
research-pipeline watch

# Specify a custom queries file
research-pipeline watch --queries ~/my-watch-queries.json

# Look back 14 days
research-pipeline watch --lookback 14

# Save results to a file
research-pipeline watch -o new-papers.json
```

The default queries file is `~/.cache/research-pipeline/watch/watch_queries.json`.
Create it with your query terms to start monitoring.

**Global paper index** (incremental deduplication across runs):

Enable in `config.toml`:

```toml
[incremental]
enabled = true
```

Then manage the index:

```bash
# List all indexed papers
research-pipeline index --list

# Full-text search the index
research-pipeline index --search "attention mechanism"

# Search with result limit
research-pipeline index --search "transformer" --search-limit 20

# Garbage collect stale entries
research-pipeline index --gc
```

**Paper clustering** (group screened papers by topic):

```bash
research-pipeline cluster --run-id <RUN_ID>
research-pipeline cluster --run-id <RUN_ID> --threshold 0.2
```

**Citation context extraction** (for reference analysis):

```bash
research-pipeline cite-context --run-id <RUN_ID>
research-pipeline cite-context --run-id <RUN_ID> --window 2
```

---

## 9. Troubleshooting

### 9.1 No results from search

**Symptoms:** "0 candidates found" or very few results.

**Solutions:**

1. **Broaden the search window:**
   ```toml
   [search]
   primary_months = 24   # search up to 2 years back
   ```

2. **Enable more sources:**
   ```bash
   research-pipeline run "topic" --source all
   ```

3. **Lower the screening threshold:**
   ```toml
   [screen]
   final_score_threshold = 0.40
   cheap_top_k = 100
   ```

4. **Use the `deep` profile**, which searches more variants:
   ```bash
   research-pipeline run "topic" --profile deep
   ```

5. **Rephrase your topic** — use the terminology from the field (e.g. "RLHF"
   instead of "reinforcement learning from human feedback").

### 9.2 PDF download failures

**Symptoms:** Download stage completes with some failures; missing PDFs.

**Solutions:**

1. **Retry failed downloads:**
   ```bash
   research-pipeline download --run-id <RUN_ID> --retry-failed
   ```

2. **Check arXiv availability.** Some papers may not have open-access PDFs.

3. **Reduce the download count** if hitting rate limits:
   ```toml
   [download]
   max_per_run = 10
   ```

4. **Wait and retry.** arXiv enforces a 3-second request floor; bursts may
   trigger temporary blocks.

### 9.3 Conversion errors

**Symptoms:** Markdown files are empty, truncated, or conversion stage fails.

**Solutions:**

1. **Try a different backend:**
   ```bash
   research-pipeline convert --run-id <RUN_ID> --backend pymupdf4llm
   ```

2. **Configure fallback backends:**
   ```toml
   [conversion]
   backend = "docling"
   fallback_backends = ["pymupdf4llm"]
   ```

3. **Use two-tier conversion:**
   ```bash
   research-pipeline convert-rough --run-id <RUN_ID>   # fast, all papers
   research-pipeline convert-fine --run-id <RUN_ID> --paper-ids 2401.12345  # high-quality, selected
   ```

4. **Force re-conversion:**
   ```bash
   research-pipeline convert --run-id <RUN_ID> --force
   ```

5. **Check the PDF directly:**
   ```bash
   research-pipeline convert-file problem-paper.pdf --backend pymupdf4llm
   ```

### 9.4 MCP connection issues

**Symptoms:** AI assistant cannot connect to the MCP server; tools not available.

**Solutions:**

1. **Verify the server path:**
   ```bash
   which research-pipeline
   research-pipeline mcp serve --verbose   # test it starts without error
   ```

2. **Use the full absolute path** in your MCP client config:
   ```json
   {
     "command": "/home/user/.local/bin/research-pipeline",
     "args": ["mcp", "serve"]
   }
   ```

3. **Print the recommended config snippet:**
   ```bash
   research-pipeline mcp config
   ```

4. **Check for port conflicts.** The MCP server uses stdio transport, not TCP.
   It should not conflict with other services.

### 9.5 Rate limiting

**Symptoms:** Slow search, warnings about rate limiting, or HTTP 429 errors.

**Solutions:**

1. **arXiv rate limiting** is built in and cannot be disabled. It enforces a
   5-second default interval. You can reduce it to the minimum 3-second floor:
   ```toml
   [arxiv]
   min_interval_seconds = 3.0
   ```

2. **For Semantic Scholar**, get a free API key to raise the rate limit significantly:
   ```toml
   [sources]
   semantic_scholar_api_key = "your-key"   # free at semanticscholar.org  # pragma: allowlist secret
   ```

3. **For Google Scholar** (`scholarly` backend), use SerpAPI for more reliable access:
   ```toml
   [sources]
   scholar_backend = "serpapi"
   serpapi_key = "your-key"
   ```

4. **Reduce `--max-results`** if hitting source rate limits during heavy use.

### 9.6 Common error messages and fixes

| Error message | Cause | Fix |
|---------------|-------|-----|
| `Run ID not found` | Wrong run ID or wrong workspace | Check `research-pipeline inspect` for valid run IDs |
| `No completed search stage` | Running screen without a search | Run `research-pipeline search --run-id <ID>` first |
| `FileNotFoundError: source registry not found` | `--registry` path is wrong | Check the path or omit `--registry` to use defaults |
| `ModuleNotFoundError: docling` | Extra not installed | `pip install research-pipeline[docling]` |
| `ModuleNotFoundError: sentence_transformers` | Reranker not installed | `pip install research-pipeline[reranker]` |
| `scholarly not installed` | Scholar source enabled but no package | `pip install research-pipeline[scholar]` |
| `API key required` | Cloud conversion backend missing key | Set the key in config or env var (see §3.4) |
| `ValidationError` (report) | Synthesis missing required sections | Run with LLM enabled or use `--profile deep` |

---

## 10. Command Reference Summary

### Academic pipeline

| Command | Description |
|---------|-------------|
| `run "topic"` | End-to-end pipeline (all stages) |
| `plan "topic"` | Generate structured query plan |
| `search` | Search academic sources |
| `screen` | Score and shortlist candidates |
| `download` | Download PDFs |
| `convert` | Convert PDFs to Markdown |
| `convert-rough` | Fast bulk conversion (pymupdf4llm) |
| `convert-fine` | High-quality conversion for selected papers |
| `convert-file <pdf>` | Standalone PDF → Markdown (no workspace) |
| `extract` | Extract sections and chunks |
| `summarize` | Generate summaries and synthesis |
| `inspect` | Show run status and artifact paths |
| `enrich` | Fill missing abstracts from Semantic Scholar |
| `expand` | Citation graph expansion |
| `quality` | Compute quality scores (citations, venue, author) |
| `analyze` | Prepare per-paper analysis tasks |
| `analyze-claims` | Decompose summaries into atomic claims |
| `score-claims` | Score confidence for claims |
| `feedback` | Record paper accept/reject decisions |
| `validate` | Validate report completeness |
| `compare` | Diff two pipeline runs |
| `coherence` | Multi-run knowledge coherence check |
| `consolidate` | Compress episodic memory, promote rules |
| `aggregate` | Strip rhetoric, aggregate evidence |
| `report` | Render report with a template |
| `cluster` | Cluster papers by topic similarity |
| `export-html` | Export synthesis report as HTML |
| `export-bibtex` | Export papers as BibTeX |
| `evaluate` | Validate pipeline outputs against schemas |
| `index` | Manage global paper index |
| `watch` | Check for new papers matching saved queries |
| `cite-context` | Extract citation contexts from Markdown |
| `memory-stats` | Memory tier statistics |
| `memory-episodes` | List recent episodic memories |
| `memory-search "topic"` | Search episodic memory |
| `kg-stats` | Knowledge graph statistics |
| `kg-query <id>` | Query a KG entity |
| `kg-ingest` | Ingest run results into KG |
| `eval-log` | Inspect three-channel execution logs |
| `horizon` | Compute Unified Horizon Metric |
| `rrp` | Recall/Reasoning/Presentation diagnostic |
| `blinding-audit` | Epistemic blinding audit |
| `setup` | Install AI skill, sub-agents, MCP config |
| `mcp serve` | Start MCP server (stdio transport) |
| `mcp config` | Print MCP client config snippet |

### Daily briefing (`brief` sub-app)

| Command | Description |
|---------|-------------|
| `brief run` | Full pipeline: poll → rank → generate → validate |
| `brief poll` | Fetch events from configured sources |
| `brief rank` | Deduplicate and rank events |
| `brief generate-daily` | Render daily Markdown digest |
| `brief validate` | Validate generated brief |
| `brief resume` | Resume from a specific stage |
| `brief dossier` | Generate in-depth dossier for a cluster |
| `brief weekly-synthesis` | Generate weekly trend memo |
| `brief export-obsidian` | Export to Obsidian vault |
| `brief feedback` | Record feedback on clusters/topics/sources |
| `brief preferences` | Compute/rollback preference adjustments |
| `brief topic-aliases` | Review topic alias suggestions |
| `brief compare-sources` | Compare ranked output with different registries |

### Global options (available on all commands)

| Flag | Description |
|------|-------------|
| `--verbose` / `-v` | Enable debug logging |
| `--config` / `-c` | Path to `config.toml` |
| `--workspace` / `-w` | Workspace root directory |
| `--run-id` | Specific run ID |
| `--version` / `-V` | Show version and exit |
| `--help` | Show help for any command |
