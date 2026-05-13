# API Reference: research-pipeline

This document is the complete CLI and MCP API reference for **research-pipeline**
— a deterministic, stage-based academic paper research pipeline.

---

## Table of Contents

1. [CLI Overview](#1-cli-overview)
2. [Core Research Pipeline Commands](#2-core-research-pipeline-commands)
3. [Convert Commands](#3-convert-commands)
4. [Analysis and Evaluation Commands](#4-analysis-and-evaluation-commands)
5. [Quality and Scoring Commands](#5-quality-and-scoring-commands)
6. [Memory Commands](#6-memory-commands)
7. [Paper Utility Commands](#7-paper-utility-commands)
8. [Output and Export Commands](#8-output-and-export-commands)
9. [Watch and Discovery Commands](#9-watch-and-discovery-commands)
10. [System and Administration Commands](#10-system-and-administration-commands)
11. [Brief Sub-app Commands (Daily AI Intelligence)](#11-brief-sub-app-commands-daily-ai-intelligence)
12. [MCP Server API](#12-mcp-server-api)
13. [Configuration Reference](#13-configuration-reference)
14. [Environment Variables](#14-environment-variables)

---

## 1. CLI Overview

### Invocation

```bash
research-pipeline [GLOBAL_OPTIONS] COMMAND [ARGS]
```

### Global Options

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--version` | `-V` | flag | — | Print version and exit |
| `--help` | — | flag | — | Show help and exit |

### Common Options (Most Commands)

Most commands accept these shared options:

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--verbose` | `-v` | flag | `false` | Enable debug logging |
| `--config PATH` | `-c` | path | `~/.research-pipeline.toml` | Path to config TOML file |
| `--workspace PATH` | `-w` | path | auto | Workspace root for storing runs |
| `--run-id TEXT` | — | string | auto-generated | Run identifier |

### Sub-apps

The CLI has three apps:

- **Main app** — all research pipeline commands (plan, search, screen, …)
- **`brief`** — daily AI intelligence briefing: `research-pipeline brief …`
- **`mcp`** — MCP server management: `research-pipeline mcp …`

### Output Modes

Most commands write structured artifacts to the workspace directory (`./runs/<run-id>/`).
Progress and status messages go to stderr via Python's `logging` module.
JSON/Markdown outputs are written to stage-specific subdirectories.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Error (bad input, stage failure, validation failure) |
| `2` | Invalid usage (typer argument error) |

---

## 2. Core Research Pipeline Commands

The 7-stage pipeline runs sequentially. Each stage is idempotent.

```
plan → search → screen → download → convert → extract → summarize
```

---

### `plan`

Normalize a topic into a structured query plan.

```bash
research-pipeline plan TOPIC [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `TOPIC` | string | yes | Research topic in natural language |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--verbose / -v` | flag | `false` | Enable debug logging |
| `--config PATH / -c` | path | auto | Config file path |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--run-id TEXT` | string | auto | Run identifier |

**Output** — `<workspace>/<run-id>/plan/query_plan.json`

**Example**

```bash
research-pipeline plan "local memory systems for AI agents"
research-pipeline plan "transformer attention" --run-id my-run-001
```

---

### `search`

Search configured academic paper sources in parallel.

```bash
research-pipeline search [TOPIC] [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `TOPIC` | string | no | Research topic (or use `--run-id` to resume) |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--run-id TEXT` | string | auto | Run identifier |
| `--resume` | flag | `false` | Resume from existing search results |
| `--source TEXT / -s` | string | from config | Source(s): `arxiv`, `scholar`, `semantic_scholar`, `openalex`, `dblp`, `huggingface`, `all` |

**Output** — `<workspace>/<run-id>/search/candidates.jsonl`

**Example**

```bash
research-pipeline search "AI agents" --source all
research-pipeline search --run-id my-run-001 --source arxiv,semantic_scholar
```

---

### `screen`

Score and rank search candidates by relevance.

```bash
research-pipeline screen --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with search results |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--resume` | flag | `false` | Resume from existing screening results |
| `--diversity / --no-diversity` | flag | from config | Enable MMR diversity-aware reranking |
| `--diversity-lambda FLOAT` | float | `0.3` | Balance: 0.0 = pure relevance, 1.0 = pure diversity |

**Output** — `<workspace>/<run-id>/screen/shortlist.json`

**Example**

```bash
research-pipeline screen --run-id my-run-001
research-pipeline screen --run-id my-run-001 --diversity --diversity-lambda 0.5
```

---

### `download`

Download PDFs for shortlisted candidates from arXiv.

```bash
research-pipeline download --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with screened shortlist |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--force` | flag | `false` | Re-download existing files |
| `--retry-failed` | flag | `false` | Re-attempt only previously failed downloads |

**Output** — `<workspace>/<run-id>/download/pdf/` (PDF files + `download_manifest.json`)

**Example**

```bash
research-pipeline download --run-id my-run-001
research-pipeline download --run-id my-run-001 --retry-failed
```

---

### `extract`

Extract structured sections from converted Markdown.

```bash
research-pipeline extract --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with converted Markdown |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--cross-encoder / --no-cross-encoder` | flag | auto-detect | Enable cross-encoder reranking for chunk retrieval |

**Output** — `<workspace>/<run-id>/extract/` (chunked content + index)

**Example**

```bash
research-pipeline extract --run-id my-run-001
research-pipeline extract --run-id my-run-001 --cross-encoder
```

---

### `summarize`

Generate per-paper summaries and cross-paper synthesis.

```bash
research-pipeline summarize --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with extracted content |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--output-format TEXT / -f` | string | `markdown` | Output format: `markdown`, `json`, `bibtex`, `structured-json` |
| `--step TEXT` | string | `all` | Step to run: `extraction`, `synthesis`, or `all` |

**Output** — `<workspace>/<run-id>/summarize/synthesis_report.md` + per-paper summaries

**Example**

```bash
research-pipeline summarize --run-id my-run-001
research-pipeline summarize --run-id my-run-001 -f json
research-pipeline summarize --run-id my-run-001 --step synthesis
```

---

### `run`

Run all pipeline stages end-to-end.

```bash
research-pipeline run TOPIC [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `TOPIC` | string | yes | Research topic in natural language |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--run-id TEXT` | string | auto | Run identifier |
| `--resume` | flag | `false` | Resume from existing results |
| `--source TEXT / -s` | string | from config | Search source(s) |
| `--profile TEXT / -p` | string | `standard` | Pipeline profile: `quick`, `standard`, `deep`, `auto` |
| `--ter-iterations INT` | int | `3` | Max THINK→EXECUTE→REFLECT iterations (0 = disabled) |
| `--auto-approve / --interactive` | flag | `true` | Auto-approve HITL gates or pause for review |

**Profiles**

| Profile | Stages |
|---|---|
| `quick` | plan → search → screen → summarize (abstract-only, no PDFs) |
| `standard` | Full 7-stage pipeline |
| `deep` | standard + expand + quality + claim analysis + TER loop |
| `auto` | Detect from query complexity |

**Example**

```bash
research-pipeline run "transformer architectures for time series"
research-pipeline run "LLM memory" --profile quick --source arxiv
research-pipeline run "deep learning" --profile deep --ter-iterations 5
```

---

## 3. Convert Commands

---

### `convert`

Convert downloaded PDFs to Markdown using the configured backend.

```bash
research-pipeline convert --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with downloaded PDFs |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--force` | flag | `false` | Re-convert existing files |
| `--backend TEXT / -b` | string | from config | Backend: `docling`, `marker`, `pymupdf4llm` |

**Example**

```bash
research-pipeline convert --run-id my-run-001
research-pipeline convert --run-id my-run-001 --backend marker
```

---

### `convert-rough`

Fast Tier 2 conversion of all downloaded PDFs using pymupdf4llm.

```bash
research-pipeline convert-rough --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with downloaded PDFs |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--force` | flag | `false` | Re-convert existing files |

**Example**

```bash
research-pipeline convert-rough --run-id my-run-001
```

---

### `convert-fine`

High-quality Tier 3 conversion of selected PDFs.

```bash
research-pipeline convert-fine --run-id RUN_ID --paper-ids IDS [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with downloaded PDFs |
| `--paper-ids TEXT` | string | **required** | Comma-separated arXiv IDs to convert |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--force` | flag | `false` | Re-convert existing files |
| `--backend TEXT / -b` | string | from config | Backend override |

**Example**

```bash
research-pipeline convert-fine --run-id my-run-001 --paper-ids 2401.12345,2401.67890
research-pipeline convert-fine --run-id my-run-001 --paper-ids 2401.12345 --backend docling
```

---

### `convert-file`

Convert a single PDF to Markdown (standalone, no pipeline workspace required).

```bash
research-pipeline convert-file PDF_PATH [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `PDF_PATH` | path | yes | Path to the PDF file to convert |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--output PATH / -o` | path | same dir as PDF | Output directory |
| `--backend TEXT / -b` | string | from config | Backend: `docling`, `marker`, `pymupdf4llm` |
| `--config PATH / -c` | path | auto | Config file |

**Example**

```bash
research-pipeline convert-file paper.pdf
research-pipeline convert-file paper.pdf -o ./output/ --backend marker
```

---

## 4. Analysis and Evaluation Commands

---

### `analyze`

Prepare per-paper analysis tasks or validate collected results.

```bash
research-pipeline analyze --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with converted papers |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--collect` | flag | `false` | Validate collected analysis JSON files (instead of generating prompts) |
| `--paper-ids TEXT` | string | all papers | Comma-separated arXiv IDs to analyze |

**Example**

```bash
research-pipeline analyze --run-id my-run-001
research-pipeline analyze --run-id my-run-001 --collect
research-pipeline analyze --run-id my-run-001 --paper-ids 2401.12345,2401.67890
```

---

### `analyze-claims`

Decompose paper summaries into atomic claims with evidence classification.

```bash
research-pipeline analyze-claims --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with paper summaries |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

**Output** — `<workspace>/<run-id>/summarize/claims/` (JSONL with classified claims)

Claims are classified as: `supported`, `partial`, `conflicting`, `inconclusive`, or `unsupported`.

**Example**

```bash
research-pipeline analyze-claims --run-id my-run-001
```

---

### `score-claims`

Score confidence for decomposed claims.

```bash
research-pipeline score-claims [--run-id RUN_ID] [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT / -r` | string | auto (latest) | Run ID with claim decompositions |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

**Example**

```bash
research-pipeline score-claims --run-id my-run-001
```

---

### `confidence-layers`

Score claims through the 4-layer confidence architecture (L1→L2→L3→L4).

```bash
research-pipeline confidence-layers [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT / -r` | string | auto | Run ID with claim decompositions |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace directory |
| `--l4-threshold FLOAT` | float | `0.50` | Confidence below which L4 verification triggers |
| `--damping FLOAT` | float | `0.80` | Fusion damping exponent (0-1; lower = more conservative) |
| `--calibrate` | flag | `false` | Fit Platt scaling from prior scored claims |

**Architecture layers**

| Layer | Name | Description |
|---|---|---|
| L1 | Fast signal | Rapid evidence keyword scoring |
| L2 | Adaptive granularity | BM25 retrieval + hedging detection |
| L3 | DINCO calibration | Distribution-aware score calibration |
| L4 | Selective verification | LLM verification for low-confidence claims only |

**Example**

```bash
research-pipeline confidence-layers --run-id my-run-001
research-pipeline confidence-layers --run-id my-run-001 --l4-threshold 0.4 --calibrate
```

---

### `evaluate`

Evaluate pipeline outputs against their schemas.

```bash
research-pipeline evaluate --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID to evaluate |
| `--stage TEXT / -s` | string | all | Specific stage to evaluate (`plan`, `search`, `screen`, `download`, `convert`, `extract`, `summarize`) |
| `--workspace TEXT / -w` | string | `runs` | Workspace directory |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline evaluate --run-id my-run-001
research-pipeline evaluate --run-id my-run-001 --stage screen
```

---

### `validate`

Validate a research report for completeness and quality.

```bash
research-pipeline validate [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--report PATH / -r` | path | — | Path to report Markdown file |
| `--run-id TEXT` | string | — | Run ID to find synthesis report (alternative to `--report`) |
| `--output PATH / -o` | path | — | Output path for validation JSON |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

Checks 14 required sections, confidence-level annotations, evidence citations, gap classifications, tables, Mermaid diagrams, and LaTeX formulas. Exits with code 1 on failure.

**Example**

```bash
research-pipeline validate --report report.md
research-pipeline validate --run-id my-run-001 --output validation.json
```

---

### `compare`

Compare two pipeline runs: papers, findings, gaps, confidence levels.

```bash
research-pipeline compare --run-a RUN_A --run-b RUN_B [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-a TEXT` | string | **required** | First run ID (baseline) |
| `--run-b TEXT` | string | **required** | Second run ID (latest) |
| `--output PATH / -o` | path | — | Output path for comparison JSON |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

**Example**

```bash
research-pipeline compare --run-a run-2025-01-01 --run-b run-2025-01-15
```

---

### `coherence`

Evaluate factual coherence across multiple pipeline runs.

```bash
research-pipeline coherence RUN_ID [RUN_ID ...] [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `RUN_IDS` | list of strings | yes (≥2) | Run IDs to evaluate coherence across |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--output PATH / -o` | path | — | Output path for coherence report JSON |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

**Example**

```bash
research-pipeline coherence run-A run-B run-C
```

---

### `consolidate`

Consolidate cross-run memory: compress episodes, promote rules, prune stale.

```bash
research-pipeline consolidate [RUN_IDS ...] [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `RUN_IDS` | list of strings | no | Run IDs to ingest (default: scan workspace) |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--output PATH / -o` | path | — | Output path for consolidation report JSON |
| `--dry-run` | flag | `false` | Compute metrics without modifying store |
| `--capacity INT` | int | `100` | Episode capacity before triggering consolidation |
| `--threshold FLOAT` | float | `0.8` | Fraction of capacity triggering consolidation |
| `--min-support INT` | int | `2` | Min run appearances for rule promotion |
| `--staleness-days INT` | int | `90` | Age threshold (days) for pruning |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline consolidate
research-pipeline consolidate run1 run2 run3 --dry-run
```

---

### `horizon`

Compute the Unified Horizon Metric (UHM) for a long-horizon agent run.

```bash
research-pipeline horizon --score FLOAT --achieved INT --target INT [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--score FLOAT` | float | **required** | Normalized task quality in [0, 1] |
| `--achieved INT` | int | **required** | Trajectory length completed |
| `--target INT` | int | **required** | Benchmark target horizon |
| `--difficulty FLOAT` | float | `0.5` | Task difficulty in [0, 1] |
| `--entropy-trend FLOAT` | float | `0.0` | Token-entropy slope (negative = locking) |
| `--reliability FLOAT` | float | `1.0` | Pass[k] reliability floor |
| `--output PATH / -o` | path | — | Write JSON result to this path |

**Example**

```bash
research-pipeline horizon --score 0.8 --achieved 40 --target 50
research-pipeline horizon --score 0.75 --achieved 30 --target 50 --difficulty 0.7 --reliability 0.9
```

---

### `rrp`

Recall / Reasoning / Presentation diagnostic for a synthesis report.

```bash
research-pipeline rrp --report PATH --shortlist PATH [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--report PATH / -r` | path | **required** | Path to synthesis report (`.md` or `.txt`) |
| `--shortlist PATH / -s` | path | **required** | Path to shortlist JSON with paper IDs |
| `--output PATH / -o` | path | — | Write JSON diagnostic to this path |

Operationalizes the DeepResearch Bench II finding (Theme 16): Information Recall is the primary bottleneck.

**Example**

```bash
research-pipeline rrp --report report.md --shortlist shortlist.json
```

---

### `blinding-audit`

Run epistemic blinding audit to detect LLM prior contamination.

```bash
research-pipeline blinding-audit [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | latest | Specific run ID to audit |
| `--workspace TEXT / -w` | string | `workspace` | Workspace directory |
| `--threshold FLOAT` | float | `0.4` | Contamination threshold for flagging papers |
| `--no-store` | flag | `false` | Do not persist results to SQLite |
| `--json` | flag | `false` | Output raw JSON instead of summary |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline blinding-audit --run-id my-run-001
research-pipeline blinding-audit --threshold 0.3 --json
```

---

### `dual-metrics`

Evaluate pipeline runs using Pass@k + Pass[k] dual metrics.

```bash
research-pipeline dual-metrics --query QUERY [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--query TEXT / -q` | string | **required** | Research query these runs address |
| `--run-ids TEXT` | string | auto-discover | Comma-separated run IDs to evaluate |
| `--workspace TEXT / -w` | string | `workspace` | Workspace directory |
| `--k INT` | int | `5` | Number of samples for Pass@k / Pass[k] |
| `--no-store` | flag | `false` | Do not persist results to SQLite |
| `--json` | flag | `false` | Output raw JSON instead of summary |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline dual-metrics --query "transformer attention" --run-ids r1,r2,r3
research-pipeline dual-metrics --query "LLM agents" --k 3
```

---

### `adaptive-stopping`

Evaluate adaptive retrieval stopping criteria.

```bash
research-pipeline adaptive-stopping SCORES_FILE [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `SCORES_FILE` | path | yes | JSON file with retrieval scores (list of lists, one per batch) |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--query TEXT / -q` | string | `""` | Original query for auto-classifying strategy |
| `--query-type TEXT / -t` | string | `auto` | Strategy: `recall`, `precision`, `judgment`, or `auto` |
| `--min-results INT` | int | `5` | Minimum results before stopping is considered |
| `--max-budget INT` | int | `500` | Hard budget limit on total results |
| `--relevance-threshold FLOAT` | float | `0.5` | Score threshold for a relevant result |
| `--output PATH / -o` | path | stdout | Output JSON path |

**Example**

```bash
research-pipeline adaptive-stopping scores.json --query "transformers" --query-type recall
```

---

## 5. Quality and Scoring Commands

---

### `quality`

Compute quality scores for candidate papers.

```bash
research-pipeline quality --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID with search/screen results |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

Evaluates: citation impact, venue reputation (CORE rankings), author h-index credibility, recency.

**Output** — `<workspace>/<run-id>/quality/quality_scores.json`

**Example**

```bash
research-pipeline quality --run-id my-run-001
```

---

### `cbr-lookup`

Look up past cases and recommend a research strategy (Case-Based Reasoning).

```bash
research-pipeline cbr-lookup --topic TOPIC [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--topic TEXT / -t` | string | **required** | Research topic to look up |
| `--workspace TEXT / -w` | string | `workspace` | Workspace directory |
| `--max-results INT` | int | `5` | Maximum similar cases to retrieve |
| `--min-quality FLOAT` | float | `0.0` | Minimum synthesis quality to consider |
| `--json` | flag | `false` | Output raw JSON instead of summary |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline cbr-lookup --topic "transformer architectures"
research-pipeline cbr-lookup --topic "LLM agents" --min-quality 0.5
```

---

### `cbr-retain`

Store a completed pipeline run as a CBR case for future retrieval.

```bash
research-pipeline cbr-retain --run-id RUN_ID --topic TOPIC [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Pipeline run ID to store |
| `--topic TEXT / -t` | string | **required** | Research topic for this run |
| `--workspace TEXT / -w` | string | `workspace` | Workspace directory |
| `--outcome TEXT` | string | `unknown` | Quality outcome: `excellent`, `good`, `adequate`, `poor`, `failed` |
| `--notes TEXT` | string | `""` | Free-text strategy notes |
| `--json` | flag | `false` | Output raw JSON instead of summary |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline cbr-retain --run-id my-run-001 --topic "transformers" --outcome good
```

---

### `kg-stats`

Show knowledge graph statistics.

```bash
research-pipeline kg-stats [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--db PATH` | path | default | KG database path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline kg-stats
research-pipeline kg-stats --db ./my-kg.db
```

---

### `kg-query`

Query an entity and its relations in the knowledge graph.

```bash
research-pipeline kg-query ENTITY_ID [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `ENTITY_ID` | string | yes | Entity ID to query |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--db PATH` | path | default | KG database path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline kg-query 2401.12345
```

---

### `kg-ingest`

Ingest pipeline results into the knowledge graph.

```bash
research-pipeline kg-ingest [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT / -r` | string | auto | Run ID to ingest |
| `--db PATH` | path | default | KG database path |
| `--verbose / -v` | flag | `false` | Debug logging |
| `--config PATH / -c` | path | auto | Config file |
| `--workspace PATH / -w` | path | auto | Workspace root |

**Example**

```bash
research-pipeline kg-ingest --run-id my-run-001
```

---

### `kg-quality`

Evaluate knowledge graph quality across 5 dimensions.

```bash
research-pipeline kg-quality [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--db TEXT` | string | default | Path to KG SQLite database |
| `--staleness-days FLOAT` | float | `365.0` | Threshold (days) for a triple to be stale |
| `--sample INT` | int | `0` | If > 0, run TWCS sampling and print sample |
| `--json` | flag | `false` | Output results as JSON |

Uses the three-layer composable architecture (TKDE 2022 + Text2KGBench).

**Example**

```bash
research-pipeline kg-quality
research-pipeline kg-quality --staleness-days 180 --json
```

---

## 6. Memory Commands

---

### `memory-stats`

Show memory tier statistics (working, episodic, semantic).

```bash
research-pipeline memory-stats [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--episodic-db PATH` | path | default | Episodic memory database path |
| `--kg-db PATH` | path | default | KG database path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline memory-stats
```

---

### `memory-episodes`

List recent episodic memories (past pipeline runs).

```bash
research-pipeline memory-episodes [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--limit INT / -n` | int | `10` | Max episodes to show |
| `--episodic-db PATH` | path | default | Episodic memory database path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline memory-episodes
research-pipeline memory-episodes --limit 5
```

---

### `memory-search`

Search episodic memory for past runs on a topic.

```bash
research-pipeline memory-search TOPIC [OPTIONS]
```

**Arguments**

| Name | Type | Required | Description |
|---|---|---|---|
| `TOPIC` | string | yes | Topic to search in episodic memory |

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--limit INT / -n` | int | `10` | Max results |
| `--episodic-db PATH` | path | default | Episodic memory database path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline memory-search "transformer"
research-pipeline memory-search "attention mechanism" --limit 5
```

---

## 7. Paper Utility Commands

---

### `expand`

Expand citation graph for specified papers via Semantic Scholar.

```bash
research-pipeline expand --run-id RUN_ID --paper-ids IDS [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID to store expanded candidates |
| `--paper-ids TEXT` | string | **required** | Comma-separated arXiv or S2 paper IDs |
| `--direction TEXT / -d` | string | `both` | `citations`, `references`, or `both` |
| `--limit INT` | int | `50` | Max related papers per seed paper per direction |
| `--reference-boost FLOAT` | float | `1.0` | Multiplier for backward (reference) limit |
| `--bfs-depth INT` | int | `0` | BFS expansion depth (0 = disabled; 2 = recommended) |
| `--bfs-top-k INT` | int | `10` | Max papers per BFS hop after BM25 ranking |
| `--bfs-query TEXT` | string | `""` | Comma-separated query terms for BFS BM25 pruning |
| `--bfs-budget INT` | int | `0` | Hard cap on total BFS papers (0 = no limit) |
| `--bfs-min-new INT` | int | `0` | Min new candidates per BFS hop to continue |
| `--snowball` | flag | `false` | Enable bidirectional snowball expansion |
| `--snowball-max-rounds INT` | int | `5` | Max snowball iteration rounds |
| `--snowball-max-papers INT` | int | `200` | Hard cap on total discovered papers |
| `--snowball-decay-threshold FLOAT` | float | `0.10` | Stop when relevant fraction drops below this |
| `--snowball-decay-patience INT` | int | `2` | Consecutive low-relevance rounds before stopping |
| `--verbose / -v` | flag | `false` | Debug logging |

**Expansion modes**

| Mode | When | Description |
|---|---|---|
| Single-hop | default | Direct citations/references |
| BFS | `--bfs-depth 2` | Multi-hop with BM25 pruning at each hop |
| Snowball | `--snowball` | Iterative bidirectional with budget-aware stopping |

**Example**

```bash
research-pipeline expand --run-id my-run-001 --paper-ids 2401.12345,2401.67890
research-pipeline expand --run-id my-run-001 --paper-ids 2401.12345 --bfs-depth 2 --bfs-query "transformer,attention"
research-pipeline expand --run-id my-run-001 --paper-ids 2401.12345 --snowball --bfs-query "memory,agents"
```

---

### `cluster`

Cluster papers by topic similarity using TF-IDF.

```bash
research-pipeline cluster --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Pipeline run ID |
| `--stage TEXT` | string | `screen` | Stage to cluster: `search` or `screen` |
| `--threshold FLOAT / -t` | float | `0.15` | Cosine similarity threshold (lower = fewer, larger clusters) |
| `--output TEXT / -o` | string | auto | Output JSON file path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline cluster --run-id my-run-001
research-pipeline cluster --run-id my-run-001 --threshold 0.2
```

---

### `enrich`

Enrich candidates with missing abstracts/metadata from Semantic Scholar.

```bash
research-pipeline enrich --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID to enrich candidates for |
| `--stage TEXT` | string | `candidates` | Stage to read from: `candidates` or `screened` |
| `--config TEXT` | string | `config.toml` | Config file path |
| `--log-level TEXT` | string | `INFO` | Log level |

**Example**

```bash
research-pipeline enrich --run-id my-run-001
research-pipeline enrich --run-id my-run-001 --stage screened
```

---

### `cite-context`

Extract citation contexts from converted Markdown papers.

```bash
research-pipeline cite-context --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID to extract contexts from |
| `--window INT` | int | `1` | Extra sentences before/after citation |
| `--output TEXT / -o` | string | auto | Output JSON file path |
| `--config TEXT` | string | `config.toml` | Config file path |
| `--log-level TEXT` | string | `INFO` | Log level |

**Example**

```bash
research-pipeline cite-context --run-id my-run-001
research-pipeline cite-context --run-id my-run-001 --window 2
```

---

### `feedback`

Record user feedback on screened papers to improve future screening.

```bash
research-pipeline feedback --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID whose screened papers to give feedback on |
| `--accept TEXT / -a` | string | — | Paper ID to accept (repeatable) |
| `--reject TEXT / -r` | string | — | Paper ID to reject (repeatable) |
| `--reason TEXT` | string | `""` | Optional reason for the decisions |
| `--show / -s` | flag | `false` | Show current feedback stats |
| `--adjust` | flag | `false` | Recompute adjusted BM25 weights from feedback |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline feedback --run-id my-run-001 --accept 2401.12345 --accept 2401.12346
research-pipeline feedback --run-id my-run-001 --reject 2401.12347 --reason "off-topic"
research-pipeline feedback --run-id my-run-001 --show
research-pipeline feedback --run-id my-run-001 --adjust
```

---

## 8. Output and Export Commands

---

### `export-html`

Export synthesis report as a self-contained HTML file.

```bash
research-pipeline export-html [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | `""` | Pipeline run ID (reads `synthesis_report.json`) |
| `--markdown TEXT` | string | `""` | Path to a Markdown report to convert (alternative to `--run-id`) |
| `--output TEXT / -o` | string | auto | Output HTML file path |
| `--title TEXT` | string | `Research Report` | Report title (used with `--markdown` mode) |
| `--config TEXT` | string | `config.toml` | Config file path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline export-html --run-id my-run-001
research-pipeline export-html --markdown report.md -o report.html --title "My Research"
```

---

### `export-bibtex`

Export papers from a pipeline stage as a BibTeX file.

```bash
research-pipeline export-bibtex --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Pipeline run ID |
| `--stage TEXT` | string | `screen` | Stage to export: `search`, `screen`, or `download` |
| `--output TEXT / -o` | string | auto | Output `.bib` file path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline export-bibtex --run-id my-run-001
research-pipeline export-bibtex --run-id my-run-001 --stage search -o refs.bib
```

---

### `report`

Render synthesis report using a configurable Jinja2 template.

```bash
research-pipeline report --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Pipeline run ID |
| `--template TEXT / -t` | string | `survey` | Template: `survey`, `gap_analysis`, `lit_review`, `executive` |
| `--custom-template TEXT` | string | `""` | Path to a custom Jinja2 template file |
| `--output TEXT / -o` | string | auto | Output Markdown file path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline report --run-id my-run-001
research-pipeline report --run-id my-run-001 -t gap_analysis
research-pipeline report --run-id my-run-001 --custom-template my-template.j2
```

---

### `aggregate`

Aggregate evidence from synthesis, stripping rhetoric.

```bash
research-pipeline aggregate --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Pipeline run ID |
| `--min-pointers INT` | int | `0` | Minimum evidence pointers per statement |
| `--max-words INT` | int | `50` | Maximum words per statement |
| `--similarity-threshold FLOAT` | float | `0.7` | Threshold for merging similar statements (0-1) |
| `--no-strip-rhetoric` | flag | `false` | Disable rhetoric stripping |
| `--format TEXT` | string | `text` | Output format: `text` or `json` |
| `--config TEXT` | string | `config.toml` | Config file path |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline aggregate --run-id my-run-001
research-pipeline aggregate --run-id my-run-001 --min-pointers 1 --format json
```

---

### `eval-log`

Inspect three-channel evaluation logs for a pipeline run.

```bash
research-pipeline eval-log --run-id RUN_ID [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | **required** | Run ID to inspect evaluation logs for |
| `--channel TEXT / -c` | string | `all` | Channel: `traces`, `audit`, `snapshots`, `summary`, `all` |
| `--stage TEXT / -s` | string | `""` | Filter by pipeline stage |
| `--limit INT / -n` | int | `50` | Maximum records to display |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--verbose / -v` | flag | `false` | Debug logging |

**Channels**

| Channel | Description |
|---|---|
| `traces` | Execution flow (JSONL) — timing, causality |
| `audit` | Structured SQLite DB — who/what/when records |
| `snapshots` | Filesystem state captures at stage boundaries |
| `summary` | Overview of all three channels |

**Example**

```bash
research-pipeline eval-log --run-id my-run-001
research-pipeline eval-log --run-id my-run-001 --channel traces --stage screen
research-pipeline eval-log --run-id my-run-001 --channel summary
```

---

## 9. Watch and Discovery Commands

---

### `watch`

Check for new papers matching saved watch queries on arXiv.

```bash
research-pipeline watch [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--queries TEXT` | string | `~/.cache/research-pipeline/watch/watch_queries.json` | Path to JSON file with saved queries |
| `--lookback INT` | int | `7` | Days to look back on first run |
| `--max-results INT` | int | `20` | Max results per query |
| `--output TEXT / -o` | string | stdout | Output JSON file |
| `--config TEXT` | string | `config.toml` | Config file path |
| `--log-level TEXT` | string | `INFO` | Log level |

**Example**

```bash
research-pipeline watch
research-pipeline watch --lookback 14 --queries my-queries.json
```

---

## 10. System and Administration Commands

---

### `inspect`

Show run status, artifacts, and cached data.

```bash
research-pipeline inspect [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id TEXT` | string | — | Specific run to inspect (omit to list all runs) |
| `--workspace PATH / -w` | path | auto | Workspace root |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline inspect
research-pipeline inspect --run-id my-run-001
```

---

### `index`

Manage the global paper index for incremental runs.

```bash
research-pipeline index [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--list` | flag | `false` | List indexed papers |
| `--gc` | flag | `false` | Garbage collect stale entries |
| `--search TEXT` | string | — | Full-text search across paper titles and abstracts |
| `--search-limit INT` | int | `50` | Max results for `--search` |
| `--db-path TEXT` | string | default | Path to index database |
| `--verbose / -v` | flag | `false` | Debug logging |

**Example**

```bash
research-pipeline index --list
research-pipeline index --search "transformer attention"
research-pipeline index --gc
```

---

### `setup`

Install skills, agents, and MCP config for AI assistant discovery.

```bash
research-pipeline setup [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--skill-target TEXT` | string | `~/.claude/skills/research-pipeline` and `~/.codex/skills/research-pipeline` | Target directory for skill installation |
| `--agents-target TEXT` | string | `~/.claude/agents` | Target directory for agent files |
| `--mcp-config-target TEXT` | string | `~/.config/research-pipeline/mcp.json` | Target file for MCP config snippet |
| `--symlink / -s` | flag | `false` | Create symlinks instead of copying |
| `--force / -f` | flag | `false` | Overwrite existing files/directories |
| `--skip-skill` | flag | `false` | Skip skill installation |
| `--skip-agents` | flag | `false` | Skip agent installation |
| `--skip-mcp` | flag | `false` | Skip MCP config snippet installation |
| `--verbose / -v` | flag | `false` | Debug logging |

**Installs**

- Skill → `~/.claude/skills/research-pipeline/` and `~/.codex/skills/research-pipeline/`
- Sub-agents → `~/.claude/agents/` (paper-analyzer, paper-screener, paper-synthesizer)
- MCP config snippet → `~/.config/research-pipeline/mcp.json`

**Example**

```bash
research-pipeline setup
research-pipeline setup --symlink --force
research-pipeline setup --skip-agents
```

---

### `version`

Print the version and exit.

```bash
research-pipeline --version
research-pipeline -V
```

---

### `mcp serve`

Run the MCP server over stdio (Model Context Protocol).

```bash
research-pipeline mcp serve [OPTIONS]
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--verbose / -v` | flag | `false` | Enable debug logging before starting the server |

**Example**

```bash
research-pipeline mcp serve
research-pipeline mcp serve --verbose
```

---

### `mcp config`

Print a reusable MCP client configuration snippet.

```bash
research-pipeline mcp config
```

Outputs a JSON snippet for configuring MCP clients (Claude Desktop, etc.) to connect to the server via stdio.

---

## 11. Brief Sub-app Commands (Daily AI Intelligence)

All brief commands use the prefix `research-pipeline brief …`.

Common options shared by most brief commands:

| Option | Type | Default | Description |
|---|---|---|---|
| `--workspace PATH / -w` | path | `./workspace` | Workspace root |
| `--date TEXT` | string | today (UTC) | Briefing date in `YYYY-MM-DD` format |
| `--verbose / -v` | flag | `false` | Enable debug logging |

---

### `brief poll`

Poll configured briefing sources and write raw/normalized artifacts.

```bash
research-pipeline brief poll [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--registry PATH / -r` | path | default | Path to briefing source registry JSON/TOML |
| `--fixture-base-dir PATH` | path | — | Base directory for registry fixture_path values |

**Example**

```bash
research-pipeline brief poll
research-pipeline brief poll --registry my-sources.json --date 2025-01-15
```

---

### `brief rank`

Deduplicate and deterministically rank normalized events.

```bash
research-pipeline brief rank [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--registry PATH / -r` | path | auto | Source registry |
| `--no-memory` | flag | `false` | Disable topic memory fatigue |
| `--no-feedback` | flag | `false` | Disable feedback weights |

**Example**

```bash
research-pipeline brief rank
research-pipeline brief rank --no-memory --date 2025-01-15
```

---

### `brief generate-daily`

Generate the daily Markdown brief from ranked clusters.

```bash
research-pipeline brief generate-daily [OPTIONS]
```

**Example**

```bash
research-pipeline brief generate-daily
research-pipeline brief generate-daily --date 2025-01-15
```

---

### `brief validate`

Validate the generated daily brief.

```bash
research-pipeline brief validate [OPTIONS]
```

Checks sections, budgets, duplicate titles, and evidence links. Exits with code 1 on failure.

**Example**

```bash
research-pipeline brief validate
research-pipeline brief validate --date 2025-01-15
```

---

### `brief run`

Run poll, rank, generate-daily, and validate in order.

```bash
research-pipeline brief run [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--registry PATH / -r` | path | default | Path to briefing source registry |
| `--fixture-base-dir PATH` | path | — | Base directory for registry fixtures |

**Example**

```bash
research-pipeline brief run
research-pipeline brief run --registry my-sources.json
```

---

### `brief dossier`

Generate a manual hot-topic dossier for one ranked cluster.

```bash
research-pipeline brief dossier [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--cluster TEXT` | string | — | Cluster ID to expand |
| `--auto` | flag | `false` | Auto-select top dossier candidates |
| `--max-count INT` | int | `1` | Max automatic dossiers |

**Example**

```bash
research-pipeline brief dossier --cluster cluster_abc123
research-pipeline brief dossier --auto --max-count 3
```

---

### `brief export-obsidian`

Export daily, topic, and source notes to an Obsidian vault.

```bash
research-pipeline brief export-obsidian --vault PATH [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--vault PATH` | path | **required** | Configured Obsidian vault root |
| `--registry PATH / -r` | path | auto | Source registry |
| `--dry-run` | flag | `false` | Compute target paths without writing notes |

**Example**

```bash
research-pipeline brief export-obsidian --vault ~/Documents/vault
research-pipeline brief export-obsidian --vault ~/vault --dry-run
```

---

### `brief feedback`

Record explicit feedback on a cluster, topic, source, event, or dossier.

```bash
research-pipeline brief feedback --signal SIGNAL [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--signal TEXT` | string | **required** | Feedback signal |
| `--cluster TEXT` | string | — | Cluster ID target |
| `--topic TEXT` | string | — | Topic ID target |
| `--source TEXT` | string | — | Source ID target |
| `--event TEXT` | string | — | Event ID target |
| `--dossier TEXT` | string | — | Dossier ID target |
| `--reason TEXT` | string | `""` | Optional feedback reason |
| `--strength FLOAT` | float | `1.0` | Signal strength (0-5) |
| `--show` | flag | `false` | List recorded feedback instead of recording |
| `--conflicts` | flag | `false` | List conflicting feedback |

**Example**

```bash
research-pipeline brief feedback --signal upvote --cluster cluster_abc123
research-pipeline brief feedback --signal downvote --topic topic_llm --reason "too broad"
research-pipeline brief feedback --show
```

---

### `brief compare-sources`

Compare ranked output with and without an expanded source registry.

```bash
research-pipeline brief compare-sources --base-registry PATH --expanded-registry PATH --date DATE [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--base-registry PATH` | path | **required** | Base source registry |
| `--expanded-registry PATH` | path | **required** | Registry with candidate source enabled |
| `--date TEXT` | string | **required** | Briefing date `YYYY-MM-DD` |
| `--fixture-base-dir PATH` | path | — | Base directory for fixtures |

**Example**

```bash
research-pipeline brief compare-sources \
  --base-registry base.json \
  --expanded-registry expanded.json \
  --date 2025-01-15
```

---

### `brief weekly-synthesis`

Generate a lightweight weekly trend memo from daily briefing reports.

```bash
research-pipeline brief weekly-synthesis --week WEEK_ID [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--week TEXT` | string | **required** | Week ID, e.g. `2026-W18` |
| `--output PATH / -o` | path | auto | Output Markdown path |

**Example**

```bash
research-pipeline brief weekly-synthesis --week 2026-W18
research-pipeline brief weekly-synthesis --week 2026-W18 -o weekly.md
```

---

### `brief resume`

Resume a briefing workflow from a specific stage.

```bash
research-pipeline brief resume --from-stage STAGE --date DATE [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--from-stage TEXT` | string | **required** | Stage to resume from: `rank`, `generate-daily`, or `validate` |
| `--date TEXT` | string | **required** | Briefing date `YYYY-MM-DD` |
| `--registry PATH / -r` | path | auto | Source registry |

**Example**

```bash
research-pipeline brief resume --from-stage generate-daily --date 2025-01-15
```

---

### `brief topic-aliases`

List, approve, or reject reviewable topic alias suggestions.

```bash
research-pipeline brief topic-aliases [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--approve TEXT` | string | — | Approve suggestion ID |
| `--reject TEXT` | string | — | Reject suggestion ID |
| `--review TEXT` | string | `""` | Review note |
| `--all` | flag | `false` | Show all statuses (not only pending) |

**Example**

```bash
research-pipeline brief topic-aliases
research-pipeline brief topic-aliases --approve suggestion_abc123
```

---

### `brief preferences`

Compute or rollback reversible preference adjustments from feedback.

```bash
research-pipeline brief preferences [OPTIONS]
```

**Additional Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--rollback TEXT` | string | — | Rollback a preference adjustment ID |
| `--min-feedback INT` | int | `3` | Minimum feedback events before adjustment |

**Example**

```bash
research-pipeline brief preferences
research-pipeline brief preferences --rollback adjustment_abc123
```

---

## 12. MCP Server API

### Overview

The MCP server exposes the full pipeline functionality over the Model Context Protocol (MCP), enabling AI assistants and tools to drive research workflows.

**Protocol**: MCP over stdio
**Transport**: JSON-RPC via standard I/O
**Start server**: `research-pipeline mcp serve`

**Capabilities**: 60 tools, 21 resources (URI templates), 6 prompts, auto-completions.

### Tool Result Envelope

All tools return a standard JSON envelope:

```json
{
  "success": true,
  "message": "Human-readable summary",
  "artifacts": { "key": "path/or/value" }
}
```

---

### MCP Tools by Category

#### Core Pipeline Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_plan_topic` | Create a structured query plan from a natural language topic | `topic`, `workspace`, `run_id` |
| `tool_search` | Search configured academic paper sources | `topic`, `workspace`, `run_id`, `resume`, `source` |
| `tool_screen_candidates` | Score and rank candidates by relevance | `workspace`, `run_id`, `resume` |
| `tool_download_pdfs` | Download PDFs for shortlisted candidates | `workspace`, `run_id`, `force` |
| `tool_convert_pdfs` | Convert downloaded PDFs to Markdown | `workspace`, `run_id`, `force`, `backend` |
| `tool_extract_content` | Extract structured sections from Markdown | `workspace`, `run_id` |
| `tool_summarize_papers` | Generate per-paper summaries and cross-paper synthesis | `workspace`, `run_id` |
| `tool_run_pipeline` | Run the full pipeline end-to-end | `topic`, `workspace`, `run_id`, `resume` |
| `tool_research_workflow` | Harness-engineered research workflow with sampling/elicitation | `topic`, `workspace`, `run_id`, `system_building`, `source`, `max_iterations`, `resume` |

#### Convert Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_convert_file` | Convert a single PDF to Markdown (no workspace) | `pdf_path`, `output_dir`, `backend` |
| `tool_convert_rough` | Fast Tier 2 conversion using pymupdf4llm | `workspace`, `run_id`, `force` |
| `tool_convert_fine` | High-quality Tier 3 conversion of selected papers | `workspace`, `run_id`, `paper_ids`, `force`, `backend` |
| `tool_list_backends` | List available PDF-to-Markdown converter backends | (none) |

#### Analysis and Evaluation Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_analyze_papers` | Prepare per-paper analysis tasks or validate collected results | `workspace`, `run_id`, `collect`, `paper_ids` |
| `tool_analyze_claims` | Decompose summaries into atomic claims with evidence classification | `workspace`, `run_id` |
| `tool_score_claims` | Score confidence for decomposed claims | `workspace`, `run_id` |
| `tool_confidence_layers` | Score claims through the 4-layer confidence architecture | `run_id`, `workspace`, `l4_threshold`, `damping`, `calibrate` |
| `tool_evaluate` | Validate pipeline outputs against their schemas | `run_id`, `stage`, `workspace` |
| `tool_validate_report` | Check report completeness (14 sections, citations, gaps) | `report_path`, `workspace`, `run_id` |
| `tool_compare_runs` | Structured diff between two pipeline runs | `run_id_a`, `run_id_b`, `workspace` |
| `tool_verify_stage` | Structural verification gates for any pipeline stage | `workspace`, `run_id`, `stage` |
| `tool_blinding_audit` | Epistemic blinding audit for LLM prior contamination | `workspace`, `run_id`, `threshold`, `store_results` |
| `tool_dual_metrics` | Pass@k + Pass[k] dual-metrics evaluation | `query`, `workspace`, `run_ids`, `k`, `store_results` |
| `tool_adaptive_stopping` | Adaptive retrieval stopping criteria evaluation | `batch_scores`, `query`, `query_type`, `min_results`, `max_budget`, `relevance_threshold` |
| `tool_horizon_metric` | Compute the Unified Horizon Metric (UHM) | `normalized_score`, `achieved_steps`, `target_steps`, `difficulty`, `entropy_trend`, `reliability` |
| `tool_rrp_diagnostic` | Recall/Reasoning/Presentation diagnostic for synthesis reports | `report_text`, `shortlist_ids` |
| `tool_coherence` | Multi-session coherence evaluation across runs | `run_ids`, `workspace` |
| `tool_consolidation` | Consolidate cross-run memory (episodes→rules) | `run_ids`, `workspace`, `dry_run`, `capacity`, `threshold`, `min_support` |

#### Quality and Scoring Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_evaluate_quality` | Compute composite quality scores for candidate papers | `workspace`, `run_id` |
| `tool_get_venue_tier` | Look up CORE venue tier and quality score | `venue_name`, `data_path` |
| `tool_compute_semantic_scores` | Compute SPECTER2 semantic similarity scores | `topic`, `workspace`, `run_id`, `model_name`, `batch_size` |
| `tool_cbr_lookup` | Look up similar past cases and recommend a strategy | `topic`, `workspace`, `max_results`, `min_quality` |
| `tool_cbr_retain` | Store a completed run as a CBR case | `run_id`, `topic`, `workspace`, `outcome`, `strategy_notes` |
| `tool_kg_stats` | Knowledge graph entity/triple statistics | `db_path` |
| `tool_kg_query` | Query entity + relations in the knowledge graph | `entity_id`, `db_path` |
| `tool_kg_ingest` | Ingest pipeline results into the knowledge graph | `workspace`, `run_id`, `db_path` |
| `tool_kg_quality` | Evaluate knowledge graph quality across 5 dimensions | `db_path`, `staleness_days`, `sample_size` |

#### Memory Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_memory_stats` | Memory tier statistics | `episodic_db`, `kg_db` |
| `tool_memory_episodes` | List recent episodic memories | `limit`, `episodic_db` |
| `tool_memory_search` | Search episodic memory by topic | `topic`, `limit`, `episodic_db` |

#### Paper Utility Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_expand_citations` | Expand citation graph for papers via Semantic Scholar | `workspace`, `run_id`, `paper_ids`, `direction`, `limit`, `snowball` |
| `tool_cluster` | Cluster papers by topic similarity using TF-IDF | `workspace`, `run_id`, `stage`, `threshold`, `output` |
| `tool_enrich` | Enrich candidates with missing metadata from Semantic Scholar | `workspace`, `run_id`, `stage`, `config_path` |
| `tool_cite_context` | Extract citation contexts from converted Markdown | `workspace`, `run_id`, `window`, `output`, `config_path` |
| `tool_record_feedback` | Record accept/reject feedback on screened papers | `workspace`, `run_id`, `accept`, `reject`, `reason`, `show`, `adjust` |
| `tool_watch` | Check for new papers matching saved watch queries | `queries`, `lookback`, `max_results`, `output`, `config_path` |
| `tool_manage_index` | Manage the global paper index | `list_papers`, `gc`, `db_path` |

#### Output and Export Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `tool_export_html` | Render synthesis report as self-contained HTML | `workspace`, `run_id`, `markdown_file`, `title`, `output` |
| `tool_export_bibtex` | Export papers from a stage as BibTeX | `workspace`, `run_id`, `stage`, `output` |
| `tool_report` | Render synthesis report using a configurable template | `workspace`, `run_id`, `template`, `custom_template`, `output` |
| `tool_aggregate_evidence` | Aggregate evidence, stripping rhetoric | `workspace`, `run_id`, `min_pointers`, `max_words`, `similarity_threshold`, `strip_rhetoric`, `output_format` |
| `tool_query_eval_log` | Query three-channel evaluation logs | `workspace`, `run_id`, `channel`, `stage`, `limit` |
| `tool_get_run_manifest` | Inspect a run's manifest | `workspace`, `run_id` |
| `tool_model_routing_info` | Show phase-aware model routing configuration | `config_path` |
| `tool_gate_info` | Show HITL gate configuration | `config_path` |

#### Brief (Daily AI Intelligence) Tools

| Tool | Description | Key Inputs |
|---|---|---|
| `brief_poll_sources` | Poll configured daily AI intelligence sources | `workspace`, `date`, `registry_path`, `fixture_base_dir` |
| `brief_rank_events` | Deduplicate and rank normalized daily intelligence events | `workspace`, `date`, `registry_path`, `use_memory`, `use_feedback` |
| `brief_generate_daily` | Generate the daily AI intelligence Markdown brief | `workspace`, `date` |
| `brief_validate_report` | Validate daily brief sections, budgets, and evidence links | `workspace`, `date` |
| `brief_run` | Run poll, rank, generate, and validate in order | `workspace`, `date`, `registry_path`, `fixture_base_dir` |
| `brief_export_obsidian` | Export daily, topic, and source notes to an Obsidian vault | `workspace`, `date`, `vault_path`, `registry_path` |
| `brief_record_feedback` | Record explicit local feedback for briefing ranking | `workspace`, `date`, `target_type`, `target_id`, `signal`, `reason`, `strength` |
| `brief_generate_dossier` | Generate a hot-topic dossier from a ranked briefing cluster | `workspace`, `date`, `cluster_id` |
| `brief_weekly_synthesis` | Generate a weekly trend memo from daily briefs | `workspace`, `week`, `output_path` |

---

### MCP Resources

Resources are read via URI templates. All return JSON or Markdown text unless stated.

| URI Template | Name | MIME Type | Description |
|---|---|---|---|
| `runs://list` | `run_list` | `application/json` | List all pipeline runs with metadata |
| `runs://{run_id}/manifest` | `run_manifest` | `application/json` | Run metadata: stages, artifacts, timing |
| `runs://{run_id}/plan` | `query_plan` | `application/json` | Structured query plan with search terms |
| `runs://{run_id}/candidates` | `candidates` | `application/jsonl` | Search candidates (multi-source metadata) |
| `runs://{run_id}/shortlist` | `shortlist` | `application/json` | Screened shortlist of relevant papers |
| `runs://{run_id}/papers/{paper_id}` | `paper_pdf` | `application/pdf` | Downloaded paper PDF |
| `runs://{run_id}/markdown/{paper_id}` | `paper_markdown` | `text/markdown` | Converted paper Markdown |
| `runs://{run_id}/summary/{paper_id}` | `paper_summary` | `application/json` | Per-paper structured summary |
| `runs://{run_id}/synthesis` | `synthesis_report` | `text/markdown` | Cross-paper synthesis report |
| `runs://{run_id}/quality` | `quality_scores` | `application/json` | Composite quality evaluation scores |
| `config://current` | `current_config` | `application/toml` | Active pipeline configuration |
| `index://papers` | `global_index` | `application/json` | Global paper index for cross-run dedup |
| `briefings://list` | `briefing_list` | `application/json` | List daily intelligence briefing runs |
| `briefings://{date}/daily` | `briefing_daily` | `text/markdown` | Daily AI intelligence Markdown brief |
| `briefings://{date}/ranked` | `briefing_ranked_clusters` | `application/jsonl` | Ranked daily intelligence clusters |
| `briefings://{date}/telemetry` | `briefing_telemetry` | `application/jsonl` | Daily intelligence telemetry |
| `briefings://{date}/validation` | `briefing_validation` | `application/json` | Daily intelligence validation result |
| `briefings://{date}/state` | `briefing_workflow_state` | `application/json` | Replayable daily intelligence workflow state |
| `workflow://{run_id}/state` | `workflow_state` | `application/json` | Workflow state: stage statuses, execution log, iterations |
| `workflow://{run_id}/telemetry` | `workflow_telemetry` | `application/jsonl` | Three-surface workflow telemetry (cognitive/operational/contextual) |
| `workflow://{run_id}/budget` | `workflow_budget` | `application/json` | Context budget usage for a workflow run |

---

### MCP Prompts

Prompts generate structured message sequences to guide AI conversations.

| Name | Arguments | Description |
|---|---|---|
| `research_topic` | `topic: str` | Full research workflow guidance: drives plan→search→screen→download→convert→extract→summarize |
| `research_workflow` | `topic: str` | Harness-engineered workflow guidance: explains 6-layer architecture, sampling, elicitation gates, and iterative synthesis |
| `analyze_paper` | `run_id: str`, `paper_id: str` | Analyze a specific converted paper for methodology, findings, and limitations |
| `compare_papers` | `run_id: str` | Compare all papers in a run: themes, contradictions, gaps, and rankings |
| `refine_search` | `run_id: str` | Refine search based on current results: query terms, sources, screening criteria |
| `quality_assessment` | `run_id: str` | Assess paper quality: interpret scores, rank papers, identify concerns |

---

## 13. Configuration Reference

Copy `config.example.toml` to `config.toml` and adjust as needed.

### Top-level

| Key | Type | Default | Description |
|---|---|---|---|
| `profile` | string | `"standard"` | Pipeline execution profile: `quick`, `standard`, `deep`, or `auto` |
| `ter_max_iterations` | int | `3` | Max THINK→EXECUTE→REFLECT iterations (0 = disabled) |
| `memory_working_capacity` | int | `50` | Max items in working memory per stage |

### `[arxiv]`

| Key | Type | Default | Description |
|---|---|---|---|
| `base_url` | string | `"https://export.arxiv.org/api/query"` | arXiv API base URL |
| `min_interval_seconds` | float | `5.0` | Minimum seconds between arXiv requests |
| `single_connection` | bool | `true` | Use a single connection to arXiv |
| `default_page_size` | int | `100` | Default results per arXiv page |
| `max_page_size` | int | `500` | Maximum results per arXiv page |
| `daily_query_cache` | bool | `true` | Cache query results for the day |

### `[cache]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable response caching |
| `search_snapshot_ttl_hours` | int | `24` | Cache TTL in hours for search snapshots |
| `cache_dir` | string | `"~/.cache/arxiv-paper-pipeline"` | Cache directory |

### `[conversion]`

| Key | Type | Default | Description |
|---|---|---|---|
| `backend` | string | `"docling"` | Conversion backend: `docling`, `marker`, `pymupdf4llm`, `mineru`, `mathpix`, `datalab`, `llamaparse`, `mistral_ocr`, `openai_vision` |
| `fallback_backends` | list of strings | `[]` | Ordered fallback backends when primary fails |
| `timeout_seconds` | int | `300` | Per-file conversion timeout |
| `rough_max_workers` | int | `4` | Parallel workers for `convert-rough` |
| `fine_max_workers` | int | `2` | Parallel workers for `convert-fine` |

#### `[conversion.datalab]`

| Key | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | `""` | Datalab API key (or env `RESEARCH_PIPELINE_DATALAB_API_KEY`) |
| `mode` | string | `"balanced"` | Conversion mode: `fast`, `balanced`, or `accurate` |

#### `[conversion.llamaparse]`

| Key | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | `""` | LlamaParse API key (or env `RESEARCH_PIPELINE_LLAMAPARSE_API_KEY`) |
| `tier` | string | `"agentic"` | Parse tier: `fast` (1 credit), `cost-effective` (3), `agentic` (10), `agentic-plus` (45) |

#### `[conversion.marker]`

| Key | Type | Default | Description |
|---|---|---|---|
| `force_ocr` | bool | `false` | Force OCR even for text PDFs |
| `use_llm` | bool | `false` | Enable LLM-assisted conversion |
| `llm_service` | string | `""` | LLM service name |
| `llm_api_key` | string | `""` | API key for LLM service |

#### `[conversion.mathpix]`

| Key | Type | Default | Description |
|---|---|---|---|
| `app_id` | string | `""` | Mathpix app ID (or env `RESEARCH_PIPELINE_MATHPIX_APP_ID`) |
| `app_key` | string | `""` | Mathpix app key (or env `RESEARCH_PIPELINE_MATHPIX_APP_KEY`) |

#### `[conversion.mineru]`

| Key | Type | Default | Description |
|---|---|---|---|
| `parse_method` | string | `"auto"` | Parse method: `auto`, `ocr`, or `txt` |
| `timeout_seconds` | int | `600` | Per-file timeout for MinerU |

#### `[conversion.mistral_ocr]`

| Key | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | `""` | Mistral API key (or env `RESEARCH_PIPELINE_MISTRAL_API_KEY`) |
| `model` | string | `"mistral-ocr-latest"` | Mistral OCR model name |

#### `[conversion.openai_vision]`

| Key | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | `""` | OpenAI API key (or env `RESEARCH_PIPELINE_OPENAI_API_KEY`) |
| `model` | string | `"gpt-4o"` | OpenAI vision model name |

### `[download]`

| Key | Type | Default | Description |
|---|---|---|---|
| `max_per_run` | int | `20` | Maximum PDFs to download per run |

### `[incremental]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable incremental runs (dedup against previous runs) |
| `global_index_path` | string | `""` | SQLite index path (empty = `~/.cache/research-pipeline/paper_index.db`) |
| `reuse_artifacts` | bool | `true` | Symlink existing PDFs/markdown instead of re-downloading |

### `[llm]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable LLM-assisted features |
| `temperature` | float | `0` | Sampling temperature |
| `profile` | string | `"default"` | LLM profile |
| `provider` | string | `"ollama"` | LLM provider: `ollama` or `openai` |
| `base_url` | string | `""` | Provider URL (empty = provider default) |
| `api_key` | string | `""` | API key (required for `openai` provider) |
| `model` | string | `""` | Model name (empty = provider default) |
| `max_tokens` | int | `4096` | Max output tokens |

### `[quality]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable quality evaluation |
| `citation_weight` | float | `0.35` | Weight for citation impact score |
| `venue_weight` | float | `0.25` | Weight for venue reputation |
| `author_weight` | float | `0.25` | Weight for author credibility (h-index) |
| `recency_weight` | float | `0.15` | Weight for recency bonus |
| `venue_data_path` | string | `""` | Custom venue rankings JSON (empty = bundled CORE data) |
| `author_cache_ttl_hours` | float | `168.0` | Author data cache TTL (7 days) |
| `min_quality_score` | float | `0.0` | Minimum quality score for CLI batch mode (0 = no filter) |

### `[screen]`

| Key | Type | Default | Description |
|---|---|---|---|
| `cheap_top_k` | int | `50` | Number of candidates for initial BM25 scoring |
| `download_top_n` | int | `8` | Number of papers to shortlist for download |
| `final_score_threshold` | float | `0.70` | Minimum BM25 score to include in shortlist |
| `llm_score_threshold` | float | `0.60` | Minimum LLM score (when LLM is enabled) |
| `use_semantic_reranking` | bool | `false` | Enable SPECTER2 semantic re-ranking |
| `embedding_model` | string | `"allenai/specter2"` | HuggingFace model for embeddings |
| `embedding_batch_size` | int | `32` | Batch size for embedding inference |
| `diversity` | bool | `false` | Enable MMR diversity-aware shortlisting |
| `diversity_lambda` | float | `0.3` | Balance: 0.0 = relevance, 1.0 = diversity |

### `[search]`

| Key | Type | Default | Description |
|---|---|---|---|
| `primary_months` | int | `6` | Primary search window in months |
| `fallback_months` | int | `12` | Fallback search window in months |
| `max_query_variants` | int | `5` | Maximum query variants to generate |
| `min_candidates` | int | `40` | Minimum candidates before stopping search |
| `min_highscore` | int | `10` | Minimum high-scoring candidates |
| `min_downloads` | int | `5` | Minimum papers to download |

### `[sources]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | list | `["arxiv"]` | Enabled sources: `arxiv`, `scholar`, `semantic_scholar`, `openalex`, `dblp`, `huggingface` |
| `scholar_backend` | string | `"scholarly"` | Google Scholar backend: `scholarly` or `serpapi` |
| `scholar_min_interval` | float | `10.0` | Seconds between scholarly requests |
| `serpapi_key` | string | `""` | Google Scholar via SerpAPI key |
| `serpapi_min_interval` | float | `5.0` | Seconds between SerpAPI requests |
| `semantic_scholar_api_key` | string | `""` | Semantic Scholar API key (optional; higher rate limits) |
| `semantic_scholar_min_interval` | float | `1.0` | Seconds between Semantic Scholar requests |
| `openalex_api_key` | string | `""` | OpenAlex API key (optional) |
| `openalex_min_interval` | float | `0.1` | Seconds between OpenAlex requests |
| `dblp_min_interval` | float | `1.0` | Seconds between DBLP requests |
| `huggingface_enabled` | bool | `true` | Enable HuggingFace daily papers source |
| `huggingface_min_interval` | float | `0.5` | Seconds between HuggingFace requests |
| `huggingface_limit` | int | `100` | Max daily papers to fetch per search |

---

## 14. Environment Variables

Environment variables override the corresponding `config.toml` entries.

| Variable | Config Key | Description |
|---|---|---|
| `RESEARCH_PIPELINE_CONFIG` | — | Path to the config TOML file |
| `RESEARCH_PIPELINE_DATALAB_API_KEY` | `conversion.datalab.api_key` | Datalab API key |
| `RESEARCH_PIPELINE_LLAMAPARSE_API_KEY` | `conversion.llamaparse.api_key` | LlamaParse API key |
| `RESEARCH_PIPELINE_MATHPIX_APP_ID` | `conversion.mathpix.app_id` | Mathpix application ID |
| `RESEARCH_PIPELINE_MATHPIX_APP_KEY` | `conversion.mathpix.app_key` | Mathpix application key |
| `RESEARCH_PIPELINE_MISTRAL_API_KEY` | `conversion.mistral_ocr.api_key` | Mistral OCR API key |
| `RESEARCH_PIPELINE_OPENAI_API_KEY` | `conversion.openai_vision.api_key` | OpenAI API key |

> **Note**: API keys for academic sources (`serpapi_key`, `semantic_scholar_api_key`, `openalex_api_key`) are set in `config.toml` only. Do not commit `config.toml` to version control — it is gitignored by default.
