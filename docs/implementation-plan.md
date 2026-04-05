# Research-Pipeline: Implementation Plan

> **Status**: Approved, not yet started
> **Created**: 2026-04-05
> **Baseline version**: v0.3.0 (8 conversion backends, FallbackConverter,
> multi-account rotation)

---

## Background

### What this project is

**research-pipeline** is a deterministic, stage-based Python 3.12+ pipeline
for searching, screening, downloading, converting, and summarizing academic
papers from arXiv and Google Scholar. It provides a Typer CLI
(`research-pipeline`) and an MCP server (stdio, 11 tools). The codebase is
~5,000 lines of Python across 40+ modules, managed by **uv**.

Current pipeline stages (v0.3.0):

```
plan → search → screen → download → convert → extract → summarize
```

Target pipeline (after this plan — progressive filtering funnel):

```
plan → search → screen/quality → download → convert-rough → convert-fine → extract → summarize
                  (Tier 1)        ────────── (Tier 2) ──────  (Tier 3) ──────────────
```

### How this project is used

This package is called by an **AI agent skill**. The AI agent handles
non-deterministic work (judgment, decisions, query formulation,
summarization), and this package handles **deterministic work** (executing
searches, computing scores, fetching metrics, downloading PDFs, converting
files). This boundary is central to all design decisions in this plan.

### What was proposed and selected

Four improvement areas were explored. The owner selected:

1. **Search quality**: Semantic re-ranking (SPECTER2), citation-graph
   expansion (Semantic Scholar), more sources (Semantic Scholar, OpenAlex,
   DBLP)
2. **Quality evaluation**: Citation metrics, venue reputation (CORE
   rankings), author credibility (h-index), composite quality score
3. **Other improvements**: Parallel conversion, retry & error recovery,
   incremental runs

**Rejected** (not in scope): LLM query expansion, multi-backend ensemble
conversion, post-conversion validation, CI/CD pipeline, export formats.

### Constraints

- **No paid services.** All external APIs must be free-tier.
- **Polite rate limiting.** Conservative request rates on all remote servers.
- **Package computes, agent decides.** The package must not autonomously
  select, filter, or discard papers. It returns computed results; the AI
  agent makes all selection/judgment decisions.

### v0.3.0 codebase state (baseline for this plan)

Before starting implementation, be aware of these recent additions:

- **5 cloud conversion backends**: Mathpix, Datalab, LlamaParse, Mistral
  OCR, OpenAI Vision (in addition to 3 local: Docling, Marker, PyMuPDF4LLM)
- **FallbackConverter** (`conversion/fallback.py`): Wraps multiple backends,
  tries in order, auto-failover on quota/rate-limit errors
- **Multi-account rotation**: Each online backend supports multiple API
  accounts via `[[conversion.<backend>.accounts]]` TOML array-of-tables
- **`_backend_kwargs_list()`**: Refactored from `_backend_kwargs()` to return
  `list[dict]` for multi-account support
- **`ConverterBackend.convert()` signature**: Now accepts `force: bool = False`
- **8 total backends** registered in `conversion/registry.py`
- **Grobid backend removed** (was a placeholder)
- **Tests added**: `test_conversion_fallback.py`,
  `test_conversion_online_backends.py`
- **MCP server** (`mcp_server/tools.py`): Still uses old `_backend_kwargs()`
  pattern; does not support fallback yet — not blocking for this plan

---

## Architecture: AI Agent ↔ Package Boundary

| Responsibility | Owner | Examples |
|---------------|-------|----------|
| **Non-deterministic** (judgment, decisions) | AI Agent | Which queries to search, which papers to expand, threshold decisions, interpreting results, summarization strategy |
| **Deterministic** (execution, computation) | Package | Execute search queries, compute BM25/semantic scores, fetch citation metrics, download PDFs, convert files |

### Design Principles for New Features

1. **Package computes, agent decides.** Every new stage should produce
   scored/enriched data and return it. The package MUST NOT autonomously
   select, filter, or discard papers. Selection is the agent's job.

2. **Fine-grained MCP tools.** Each new capability needs a corresponding
   MCP tool with explicit inputs. The agent calls tools individually and
   orchestrates the workflow. Example: `fetch_citations(paper_id)` NOT
   `expand_and_filter_top_n()`.

3. **CLI still supports batch mode.** For human CLI users, the existing
   linear pipeline (`run` command) remains. It uses config-driven defaults
   for thresholds/selection. But the MCP tools give the agent full control.

4. **No hidden decisions.** If the package must apply a threshold (e.g.,
   rate-limit caps), it should be an explicit input parameter, not a
   hardcoded policy.

### Current Autonomous Decision Points (pre-existing, NOT changed by this plan)

- **`screen` stage**: `select_topk()` picks top-k by BM25 score — this is
  deterministic given the config parameters (`cheap_top_k`, `download_top_n`,
  weights). The agent controls behavior via config. The agent skill can also
  bypass the screen stage and call scoring + selection separately.
- **`run` command**: Orchestrates all stages linearly. Kept for CLI
  convenience. The agent skill calls individual stages instead.

### New Tool Design Pattern

Each new phase exposes tools that follow this pattern:

```
Agent provides:  explicit inputs (paper IDs, queries, parameters)
Package returns: computed results (scores, metrics, candidate lists)
Agent decides:   what to do next (which papers to keep, expand, download)
```

### Progressive Filtering Funnel

The core workflow is a **3-tier progressive filter**. Each tier uses
increasingly expensive methods to evaluate papers, allowing the agent to
narrow the set at each stage:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TIER 1 — Abstract-Level Filter (cheap, no PDF download)            │
│                                                                     │
│ ~200 candidates from search                                         │
│   → BM25 scoring (existing)                                         │
│   → Semantic scoring via SPECTER2 (Phase 3)                         │
│   → Quality metrics: citations, venue tier, author h-index (Phase 5)│
│   → Agent filters to ~30-50 papers worth downloading                │
├─────────────────────────────────────────────────────────────────────┤
│ TIER 2 — Rough Markdown Filter (medium cost, fast CPU conversion)   │
│                                                                     │
│ ~30-50 papers downloaded as PDF                                     │
│   → Fast convert via pymupdf4llm (always, ~seconds per paper)       │
│   → Agent reads rough markdown (no LaTeX, basic tables)             │
│   → Agent filters to ~10-15 papers worth fine-converting            │
├─────────────────────────────────────────────────────────────────────┤
│ TIER 3 — Fine Markdown (expensive, high-quality conversion)         │
│                                                                     │
│ ~10-15 selected papers                                              │
│   → High-quality convert via configured backend (docling/marker/    │
│     cloud) with FallbackConverter                                   │
│   → Agent performs deep analysis on polished output                 │
│   → Extract stage runs on fine markdown for chunk retrieval          │
│   → Synthesis and final output                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Resource savings**: By filtering at each tier, cloud API costs drop
~70-80% (converting ~10 papers instead of ~50), and local GPU time drops
similarly. The agent's judgment at each tier ensures only the most
relevant and high-quality papers receive expensive processing.

### Agent Skill Workflow (progressive filtering)

```
Agent formulates queries
  → Package: search(topic, source)                     → candidates.jsonl

──── TIER 1: Abstract-level filter ────
Agent reviews raw candidates
  → Package: screen(run_id)                            → BM25 scores
  → Package: compute_semantic_scores(run_id, topic)    → semantic scores
  → Package: compute_quality_scores(run_id, paper_ids) → quality scores
Agent optionally expands citation graph
  → Package: fetch_citations(paper_id)                 → related papers
Agent picks ~30-50 papers to download

──── TIER 2: Rough markdown filter ────
  → Package: download(run_id, paper_ids)               → PDFs
  → Package: convert_rough(run_id)                     → rough markdown
Agent reads rough markdown, evaluates full paper content
Agent picks ~10-15 papers worth fine-converting

──── TIER 3: Fine markdown + deep analysis ────
  → Package: convert_fine(run_id, paper_ids)           → fine markdown
  → Package: extract(run_id)                           → chunks
Agent synthesizes findings from fine markdown
```

---

## External Services & Free Tier Limits

| Service | Cost | Rate Limit (Free) | Use Cases | Politeness Target |
|---------|------|-------------------|-----------|-------------------|
| **Semantic Scholar API** | Free (API key recommended) | 1 req/sec (no key); ~100 req/5min (with free key) | Search, citations, references, author metrics, venue | **1 req/sec** (no-key safe) |
| **OpenAlex API** | Free (API key required since Feb 2026) | 100K credits/day; 100 req/sec max | Search, citations, author, venue, works | **5 req/sec** (polite) |
| **DBLP API** | Free (no key) | No stated limit; fair-use policy | CS paper search | **1 req/2sec** (polite) |
| **SPECTER2** | Free (Apache 2.0, local model) | ∞ (runs locally on CPU/GPU) | Embedding-based semantic re-ranking | N/A (local) |
| **CORE Rankings** | Free (static data file) | N/A (bundled JSON) | Venue tier scoring (A*, A, B, C) | N/A (offline) |

---

## Phase 1: Infrastructure Foundations

**Goal**: Build reusable primitives that all subsequent phases depend on.

### 1.1 Generic Rate Limiter
- Extract a reusable `RateLimiter` class from `ArxivRateLimiter`
- Thread-safe, configurable `min_interval`
- Place in `src/research_pipeline/infra/rate_limit.py`
- Refactor `ArxivRateLimiter` to use/extend it (backward-compatible)
- Each new source gets its own `RateLimiter` instance

### 1.2 Retry Decorator with Exponential Backoff
- Create `src/research_pipeline/infra/retry.py`
- `@retry(max_attempts=3, backoff_base=2.0, retryable_exceptions=(RequestException, Timeout))`
- Respects `Retry-After` header (HTTP 429)
- Jitter to avoid thundering herd
- Logging at each retry attempt
- Apply retroactively to arXiv client, download, and all new sources

### 1.3 Extend CandidateRecord with Quality Fields
- Add optional fields to `CandidateRecord` (backward-compatible, all `None` default):
  - `source: str = "arxiv"` — which source found this paper
  - `doi: str | None = None`
  - `semantic_scholar_id: str | None = None`
  - `openalex_id: str | None = None`
  - `citation_count: int | None = None`
  - `influential_citation_count: int | None = None`
  - `venue: str | None = None`
  - `year: int | None = None`

### 1.4 Quality Score Model
- Create `src/research_pipeline/models/quality.py`
- `QualityScore(BaseModel)`:
  - `citation_impact: float` (0–1 normalized)
  - `venue_tier: str | None` (A*, A, B, C, or None)
  - `venue_score: float` (0–1)
  - `author_credibility: float` (0–1)
  - `reproducibility: float` (0–1, future)
  - `composite_score: float` (0–1 weighted blend)
  - `details: dict` (raw metrics for transparency)

---

## Phase 2: New Search Sources

**Goal**: Add Semantic Scholar, OpenAlex, and DBLP as search sources.
The package executes queries; the agent formulates them.

### 2.1 Semantic Scholar Search Source
- `src/research_pipeline/sources/semantic_scholar_source.py`
- Implements `SearchSource` protocol
- Uses `/paper/search` endpoint with query, year range, fields
- Rate limit: 1 req/sec (conservative, no API key needed for search)
- Optional API key via `config.sources.semantic_scholar_api_key` or
  env `RESEARCH_PIPELINE_S2_API_KEY`
- Returns `CandidateRecord` with `source="semantic_scholar"`,
  `semantic_scholar_id`, `citation_count`, `doi`
- Max 100 results per search (API limit), paginated
- Maps S2 fields → CandidateRecord (arxiv_id extracted from externalIds)

### 2.2 OpenAlex Search Source
- `src/research_pipeline/sources/openalex_source.py`
- Implements `SearchSource` protocol
- Uses `/works` endpoint with `search` filter + `from_publication_date`
- Rate limit: 5 req/sec (well within 100K credits/day)
- API key via `config.sources.openalex_api_key` or
  env `RESEARCH_PIPELINE_OPENALEX_API_KEY`
- Returns `CandidateRecord` with `source="openalex"`, `openalex_id`,
  `citation_count`, `doi`, `venue`
- Pagination via cursor
- Extracts arXiv ID from `ids.openalex` → `locations[].source`

### 2.3 DBLP Search Source
- `src/research_pipeline/sources/dblp_source.py`
- Implements `SearchSource` protocol
- Uses `/search/publ/api` endpoint (JSON format)
- Rate limit: 1 req/2sec (polite, no key needed)
- Returns `CandidateRecord` with `source="dblp"`, `doi`
- Max 1000 results per query (API cap), default 100
- Maps DBLP fields → CandidateRecord
- Note: DBLP has no abstracts; set `abstract=""` and flag for
  enrichment from other sources

### 2.4 Source Configuration
- Extend `SourcesConfig` in `config/models.py`:
  ```
  semantic_scholar_enabled: bool = False
  semantic_scholar_api_key: str = ""
  semantic_scholar_min_interval: float = 1.0
  openalex_enabled: bool = False
  openalex_api_key: str = ""
  openalex_min_interval: float = 0.2  (5 req/sec)
  dblp_enabled: bool = False
  dblp_min_interval: float = 2.0
  ```
- Update `config/defaults.py` and `config.example.toml`
- Add to `enabled` list resolution in `cmd_search.py`

### 2.5 Cross-Source Enrichment
- After multi-source search + dedup, enrich DBLP results (missing abstracts)
  by looking up the same paper on Semantic Scholar or OpenAlex via DOI
- New function: `enrich_candidates(candidates, sources_config)` in
  `sources/enrichment.py`

### 2.6 Enhanced Deduplication
- Extend `dedup_cross_source()` to also dedup by DOI
- Priority order when merging: arXiv > Semantic Scholar > OpenAlex > DBLP
  (prefer the record with the richest metadata)

### 2.7 MCP Tools (agent interface)
- The existing **`search`** MCP tool already accepts a `source` parameter.
  Extend it to accept the new source names: `"semantic_scholar"`, `"openalex"`,
  `"dblp"`. The agent chooses which sources to query.
- **`enrich_candidates`** — New MCP tool
  - Input: `run_id` (operates on candidates from search stage)
  - Behavior: Fills in missing abstracts/metadata from other sources
  - Output: Count of enriched candidates
  - Deterministic: just data fetching, no filtering

---

## Phase 3: Semantic Re-Ranking

**Goal**: Add embedding-based scoring using SPECTER2. The package computes
similarity scores; the agent decides how to use them.

### 3.1 Embedding Backend
- `src/research_pipeline/screening/embedding.py`
- Load SPECTER2 model locally (`allenai/specter2` from HuggingFace)
- Function: `compute_embeddings(texts: list[str]) -> np.ndarray`
- Batch inference for efficiency (batch_size=32)
- Model loaded lazily (first use) and cached in memory
- CPU-only by default; GPU if available via `torch.cuda.is_available()`

### 3.2 Semantic Scoring
- Function: `score_semantic(topic: str, candidates: list[CandidateRecord]) -> list[float]`
- Embed topic query (title+abstract format: `"{topic} [SEP]"`)
- Embed each candidate's `title + " " + abstract`
- Cosine similarity → normalized scores [0, 1]
- Return per-candidate semantic scores

### 3.3 Integrate into Screen Stage (opt-in, for CLI batch mode)
- In `screening/heuristic.py`, add `"semantic_similarity"` weight (default 0.0)
- When `config.screen.use_semantic_reranking = True` and SPECTER2 available:
  - Compute semantic scores
  - Blend with BM25: adjust default weights to include semantic
  - New default weights when semantic enabled:
    ```
    bm25_title: 0.20
    bm25_abstract: 0.25
    semantic_similarity: 0.25
    cat_match: 0.12
    negative_penalty: 0.08
    recency_bonus: 0.10
    ```
- Add `CheapScoreBreakdown.semantic_score: float | None` field
- **Note**: In CLI batch mode, the screen stage uses these blended scores
  with `select_topk()` as before. In agent mode, the agent can call scoring
  and selection separately (see 3.5).

### 3.4 Configuration
- Add to `ScreenConfig`:
  ```
  use_semantic_reranking: bool = False
  embedding_model: str = "allenai/specter2"
  embedding_batch_size: int = 32
  ```
- Add `specter2` optional dependency in `pyproject.toml`:
  ```
  specter2 = ["transformers>=4.30", "torch>=2.0", "adapters>=0.2"]
  ```

### 3.5 MCP Tools (agent interface)
- **`compute_semantic_scores`** — New MCP tool
  - Input: `run_id`, `topic` (or explicit query text)
  - Behavior: Loads candidates from search stage, computes SPECTER2
    similarity scores for all candidates
  - Output: List of `{arxiv_id, semantic_score}` — returns ALL scores,
    no filtering or selection
  - The agent uses these scores alongside BM25 scores to decide which
    papers to keep
- **`score_candidates`** — Enhance existing screen MCP tool to also
  return individual score breakdowns (BM25 + semantic components) so
  the agent can inspect WHY a paper scored high/low

---

## Phase 4: Citation-Graph Expansion

**Goal**: Provide tools to discover related papers via citation graph. The
agent decides WHICH papers to expand; the package fetches and returns results.

### 4.1 Citation Graph Client
- `src/research_pipeline/sources/citation_graph.py`
- Uses Semantic Scholar `/paper/{id}/citations` and `/paper/{id}/references`
- Rate limit: 1 req/sec (shared with S2 rate limiter)
- Functions:
  - `get_citations(paper_id, limit=50) -> list[CandidateRecord]`
  - `get_references(paper_id, limit=50) -> list[CandidateRecord]`

### 4.2 Expansion Logic (agent-driven, no autonomous decisions)
- The package does NOT decide which papers to expand — the agent does.
- The package provides:
  - `fetch_citations(paper_id)` — returns citing papers as CandidateRecords
  - `fetch_references(paper_id)` — returns referenced papers as CandidateRecords
  - `fetch_related(paper_ids, direction="both|citations|references")` — batch
    helper that fetches for multiple papers and deduplicates results
- All results are returned unfiltered. The agent decides:
  - Which seed papers to expand (based on screening scores, its judgment)
  - How many to expand (controls the `paper_ids` list)
  - Which expanded papers to add to the shortlist (based on returned metadata)
  - Whether expanded papers need scoring (agent can call screen/semantic tools)
- Dedup expanded results against existing candidate set (by arxiv_id, DOI, S2 ID)

### 4.3 Configuration
- Add to `SearchConfig`:
  ```
  citation_expansion_max_per_paper: int = 50
  semantic_scholar_api_key: str = ""  (shared with Phase 2)
  ```
- No `citation_expansion_top_n` — the agent decides how many papers to expand
- No `citation_expansion_enabled` — the agent calls the tool when it wants to

### 4.4 CLI Command (batch mode for human users)
- `research-pipeline expand --run-id <ID> --paper-ids <comma-separated>`
- Requires explicit paper IDs — no autonomous "pick top N" logic
- Optional: `--direction citations|references|both` (default: both)
- Output: `expand/expanded_candidates.jsonl`
- For CLI batch use with `run` command: expand is **skipped** by default.
  The agent skill calls it explicitly when needed.

### 4.5 MCP Tools (agent interface)
- **`fetch_citations`** — New MCP tool
  - Input: `paper_id` (arxiv_id or S2 paper ID), `direction`, `limit`
  - Output: List of CandidateRecords (citing/referenced papers)
  - No filtering — returns all results up to limit
- **`batch_fetch_citations`** — New MCP tool
  - Input: `paper_ids` (list), `direction`, `limit_per_paper`
  - Output: Deduplicated list of CandidateRecords across all input papers
  - Respects rate limit (1 req/sec to S2 API)
- The agent decides: "paper X has high relevance + quality → let me
  explore its citation neighborhood" — this is non-deterministic judgment

---

## Phase 5: Quality Evaluation (Tier 1 — runs BEFORE download)

**Goal**: Multi-dimensional paper quality scoring on abstract + metadata
only — no PDF download needed. This is part of Tier 1 filtering. The
package computes metrics deterministically from Semantic Scholar API data;
the agent interprets scores and decides which papers to download.

### 5.1 Citation Metrics Enrichment
- `src/research_pipeline/quality/citation_metrics.py`
- For each paper (by paper ID), fetch from Semantic Scholar:
  - `citationCount`, `influentialCitationCount`
  - `citationVelocity` (computed: citations / years_since_publication)
  - `isOpenAccess`, `fieldsOfStudy`
- Rate limit: 1 req/sec, batch endpoint (`/paper/batch`) for efficiency
  (up to 500 papers per request)
- Normalize citation count per field (CS papers vs. biology differ vastly):
  - Log-scale normalization: `min(1.0, log(1 + citations) / log(1 + 1000))`

### 5.2 Venue Reputation Scoring
- `src/research_pipeline/quality/venue_scoring.py`
- **Static tier data**: Bundle CORE rankings as a JSON lookup file in
  `src/research_pipeline/quality/data/core_rankings.json`
  - Map venue name → tier (A*, A, B, C)
  - Tier → score: A*=1.0, A=0.8, B=0.5, C=0.3, unknown=0.1
- **Dynamic lookup**: If venue not in static data, try Semantic Scholar
  venue API to get venue name, then fuzzy-match against CORE data
- **arXiv-only papers**: Score 0.1 (unreviewed preprint penalty) unless
  published venue is known from external sources

### 5.3 Author Credibility Metrics
- `src/research_pipeline/quality/author_metrics.py`
- For each paper's authors, fetch from Semantic Scholar Author API:
  - `hIndex`, `citationCount`, `paperCount`
- Aggregate per paper: use **max** author h-index (senior author signal)
- Normalize: `min(1.0, log(1 + max_h_index) / log(1 + 100))`
- Rate limit: 1 req/sec; batch by paper (look up authors of each paper)
- Cache author data (authors don't change frequently; 7-day TTL)

### 5.4 Composite Quality Score
- `src/research_pipeline/quality/composite.py`
- Compute weighted score:
  ```
  composite = w_citation * citation_impact
            + w_venue * venue_score
            + w_author * author_credibility
            + w_recency * recency_bonus
  ```
- Default weights (configurable in config.toml):
  ```
  citation_weight: 0.35
  venue_weight: 0.25
  author_weight: 0.25
  recency_weight: 0.15
  ```
- Output: `QualityScore` model per paper, stored in quality manifest
- **No filtering**: The package returns ALL scores. The agent decides
  which papers are "high quality enough" to download.

### 5.5 Quality Stage Integration
- New CLI command: `research-pipeline quality --run-id <ID>`
- Accepts explicit paper IDs OR operates on shortlist from screen stage
- Produces `quality/quality_scores.jsonl` — one QualityScore per paper
- **Does NOT decide** which papers to download. It only computes and
  persists scores. In CLI batch mode (`run` command), the download stage
  can optionally filter by a `min_quality_score` threshold. In agent mode,
  the agent reads quality scores and makes the selection.

### 5.6 MCP Tools (agent interface)
- **`compute_quality_scores`** — New MCP tool
  - Input: `run_id` + optional `paper_ids` (list of arxiv_id or S2 ID).
    If omitted, scores all candidates in the run.
  - Output: List of `QualityScore` objects with full breakdowns
    (citation_impact, venue_score, author_credibility, composite)
  - No filtering — returns all computed scores
  - The agent inspects scores and decides the final paper set
- **`get_venue_tier`** — New MCP tool (lightweight lookup)
  - Input: `venue_name`
  - Output: `{tier: "A*"|"A"|"B"|"C"|null, score: float}`
  - For quick agent inspection without running full quality pipeline

### 5.7 Configuration
- New `QualityConfig` in `config/models.py`:
  ```
  enabled: bool = False
  citation_weight: float = 0.35
  venue_weight: float = 0.25
  author_weight: float = 0.25
  recency_weight: float = 0.15
  venue_data_path: str = ""  (empty = use bundled data)
  author_cache_ttl_hours: float = 168.0  (7 days)
  min_quality_score: float = 0.0  (for CLI batch mode; 0 = no filter)
  ```

---

## Phase 6: Retry & Error Recovery

**Goal**: Make all network operations resilient to transient failures.

### 6.1 Apply Retry Decorator (from Phase 1.2)
- Wrap all HTTP calls in search sources (arXiv, Scholar, S2, OpenAlex, DBLP)
- Wrap PDF download calls
- Wrap citation graph expansion calls
- Wrap quality enrichment calls
- Default: 3 retries, exponential backoff (2s, 4s, 8s), jitter ±25%
- **Note**: `FallbackConverter` (added in v0.3.0) handles backend-level
  failover with quota/rate-limit detection. The retry decorator is
  complementary — it handles transient HTTP errors (timeouts, 5xx)
  *within* a single backend attempt, while FallbackConverter handles
  *across*-backend failover. Both layers are needed.

### 6.2 Dead-Letter Tracking
- Track permanently failed items in manifest:
  - `download_manifest.jsonl`: already has `status="failed"` + `error`
  - `convert_manifest.jsonl`: already has `status="failed"` + `error`
  - Add `retry_count: int` and `last_error: str` fields
- New CLI flag: `--retry-failed` to re-attempt only failed items

### 6.3 Graceful Degradation
- If an optional source (S2, OpenAlex, DBLP) fails entirely, log warning
  and continue with results from other sources
- If quality enrichment fails for a paper, set quality_score = None
  (don't block download/conversion)
- If SPECTER2 model fails to load, fall back to BM25-only scoring

---

## Phase 7: Two-Tier Conversion (Tier 2 + Tier 3)

**Goal**: Split conversion into two tiers aligned with the progressive
filtering funnel. Tier 2 (rough) converts ALL downloaded PDFs quickly
with pymupdf4llm so the agent can read the full text. Tier 3 (fine)
converts only agent-selected papers with a high-quality backend.

### 7.1 Rough Conversion Stage (Tier 2)
- New CLI command: `research-pipeline convert-rough --run-id <ID>`
- **Always uses pymupdf4llm** — fastest backend, CPU-only, no dependencies
- Converts ALL successfully downloaded PDFs (reads `download_manifest.jsonl`)
- Output directory: `convert_rough/` (within the run directory)
- Output manifest: `convert_rough/convert_rough_manifest.jsonl`
- Parallel execution with `ProcessPoolExecutor` (pymupdf4llm is CPU-bound):
  - Configurable: `config.conversion.rough_max_workers` (default: 4)
  - Higher default than fine because pymupdf4llm is lightweight
- This stage is **fast and cheap** — seconds per paper, no GPU or API needed
- The agent reads the rough markdown output to evaluate full paper content
  and decide which papers deserve fine conversion

### 7.2 Fine Conversion Stage (Tier 3)
- New CLI command: `research-pipeline convert-fine --run-id <ID> --paper-ids <comma-separated>`
- **Requires explicit paper IDs** — the agent selects which papers to convert
- Uses the configured backend (docling, marker, or cloud) with FallbackConverter
- Backend selection follows existing `config.conversion.backend` +
  `config.conversion.fallback_backends` logic via `_create_converter()`
- Output directory: `convert_fine/` (within the run directory)
- Output manifest: `convert_fine/convert_fine_manifest.jsonl`
- Parallel execution:
  - For local backends (docling, marker): `ProcessPoolExecutor`
    (CPU/GPU-bound), `config.conversion.fine_max_workers` (default: 2)
  - For cloud backends via FallbackConverter: `ThreadPoolExecutor`
    (I/O-bound, not pickle-safe), same worker config
  - Detect via `isinstance(converter, FallbackConverter)`
- **FallbackConverter** is only used here (not in rough conversion) — this
  is where multi-account rotation and backend failover matter

### 7.3 ConvertManifestEntry Extension
- Add `tier: Literal["rough", "fine"] = "rough"` field to
  `ConvertManifestEntry` (backward-compatible default)
- The existing `converter_name` field already distinguishes backends
  (e.g., "pymupdf4llm" vs "docling")
- For the `status` field: same 3 values (`converted`, `skipped_exists`, `failed`)

### 7.4 Extract Stage Update
- The extract stage reads fine markdown when available, falls back to rough:
  1. Read `convert_fine/convert_fine_manifest.jsonl` if it exists
  2. For papers NOT in fine manifest, read `convert_rough/convert_rough_manifest.jsonl`
  3. Prefer fine markdown because it has better equation/table rendering
- The extraction chunker is format-agnostic (only needs heading structure),
  so it works correctly with both rough and fine markdown

### 7.5 Backward Compatibility
- The existing `convert` command remains as-is — it converts all PDFs with
  the configured backend (single-tier, pre-existing behavior)
- `convert-rough` and `convert-fine` are NEW commands for the two-tier workflow
- The existing `run` command keeps using the single `convert` stage
- The agent skill uses `convert-rough` → agent filters → `convert-fine`

### 7.6 MCP Tools (agent interface)
- **`convert_rough`** — New MCP tool
  - Input: `run_id`
  - Behavior: Converts all downloaded PDFs with pymupdf4llm
  - Output: Count of converted/skipped/failed papers
  - The agent reads the rough markdown files to evaluate content
- **`convert_fine`** — New MCP tool
  - Input: `run_id`, `paper_ids` (list of arxiv_id to convert)
  - Behavior: Converts selected PDFs with configured high-quality backend
  - Output: Count of converted/skipped/failed papers
  - The agent selected these papers after reading rough markdown

### 7.7 Workspace Directory Structure
```
run_<id>/
  download/
    *.pdf
    download_manifest.jsonl
  convert_rough/                    ← Tier 2 (all papers, pymupdf4llm)
    *.md
    convert_rough_manifest.jsonl
  convert_fine/                     ← Tier 3 (selected papers, high-quality)
    *.md
    convert_fine_manifest.jsonl
  extract/                          ← reads fine markdown, falls back to rough
    *.extract.json
```

### 7.8 Configuration
- Add to `ConversionConfig`:
  ```
  rough_max_workers: int = 4     (pymupdf4llm is lightweight)
  fine_max_workers: int = 2      (docling/cloud are heavier)
  ```
- Existing `backend`, `fallback_backends`, and account configs apply to
  fine conversion only
- Rough conversion always uses pymupdf4llm (hardcoded, not configurable)

### 7.9 Progress Reporting
- Rough: "Rough-converting PDF 3/45 (2401.12345v2)..."
- Fine: "Fine-converting PDF 2/12 (2401.67890v1) [docling]..."
- Summary: "Rough: 42 converted, 3 failed | Fine: 10 converted, 2 failed"

---

## Phase 8: Incremental Runs

**Goal**: Avoid re-processing papers already handled in previous runs.

### 8.1 Global Paper Index
- `src/research_pipeline/storage/global_index.py`
- SQLite database at `~/.cache/research-pipeline/paper_index.db`
- Schema:
  ```sql
  CREATE TABLE papers (
    arxiv_id TEXT,
    doi TEXT,
    s2_id TEXT,
    title TEXT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    pdf_path TEXT,
    markdown_path TEXT,
    summary_path TEXT,
    pdf_sha256 TEXT,
    indexed_at TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_id)
  );
  CREATE INDEX idx_doi ON papers(doi);
  CREATE INDEX idx_s2_id ON papers(s2_id);
  CREATE INDEX idx_title ON papers(title);
  ```

### 8.2 Dedup Against Previous Runs
- During search: check global index for existing papers
- Flag them in `CandidateRecord` as `previously_processed: bool`
- During screen: optionally skip previously processed papers
  (configurable: `config.search.skip_previously_processed: bool = False`)

### 8.3 Artifact Linking
- During download: if PDF already exists in another run, symlink or copy
  rather than re-downloading
- During convert: if markdown already exists, reuse it
- Lookup: `global_index.find_artifact(arxiv_id, stage) -> Path | None`

### 8.4 Index Maintenance
- Auto-register completed artifacts after each stage
- CLI command: `research-pipeline index --list` to browse the global index
- CLI command: `research-pipeline index --gc` to clean stale entries
  (referenced files no longer exist)

### 8.5 Configuration
- Add to `PipelineConfig`:
  ```
  incremental_enabled: bool = False
  global_index_path: str = ""  (empty = default cache location)
  reuse_artifacts: bool = True  (symlink existing PDFs/markdown)
  ```

---

## Phase Dependency Graph

```
Phase 1 (Infra)
  ├── Phase 6 (Retry) ──→ applies to all phases
  ├── Phase 2 (New Sources)
  │     └── Phase 4 (Citation Graph) ──→ uses S2 client from 2.1
  ├── Phase 3 (Semantic Re-Ranking)
  ├── Phase 5 (Quality Eval) ──→ uses S2 client from 2.1
  ├── Phase 7 (Two-Tier Conversion) ──→ independent of 2-5
  └── Phase 8 (Incremental Runs) ──→ independent
```

## Implementation Order

```
Phase 1  →  Phase 6  →  Phase 2  →  Phase 3
                          ↓
                        Phase 4  →  Phase 5
                                      ↓
                              Phase 7  →  Phase 8
```

- Phase 1 first (all others depend on it)
- Phase 6 early (retry benefits all subsequent phases)
- Phase 2 before 4 and 5 (they reuse S2 client)
- Phase 7 after Phase 5: quality scoring + rough conversion form the
  Tier 1→2 transition in the progressive filtering funnel
- Phase 8 last (incremental runs benefit from all stages being complete)

---

## New Pipeline Stages (Final)

**CLI batch mode** (for human users — linear, config-driven):
```
plan → search → screen → [expand] → quality → download → convert → extract → summarize
                           ↑          ↑                     ↑
                        (new)      (new, Tier 1)     (existing, single-tier)
```

Note: The CLI `run` command keeps using the existing single `convert` stage
for backward compatibility. The two-tier conversion (`convert-rough` +
`convert-fine`) is designed for the agent skill workflow.

**Agent skill mode** (progressive filtering funnel):
```
──── TIER 1: Abstract-level filter (no PDF needed) ────
Agent: formulate queries
  Package: search(topic, source)                       → candidates.jsonl
Agent: review raw candidates
  Package: screen(run_id)                              → BM25 scored candidates
  Package: compute_semantic_scores(run_id, topic)      → semantic scores
Agent: optionally expand citation graph
  Package: fetch_citations(paper_id)                   → related papers
  Package: batch_fetch_citations(paper_ids)            → related papers (batch)
Agent: evaluate quality on abstracts + metadata
  Package: compute_quality_scores(run_id, paper_ids)   → quality scores
Agent picks ~30-50 papers to download (Tier 1 → Tier 2 transition)

──── TIER 2: Rough markdown filter (fast pymupdf4llm) ────
  Package: download(run_id, paper_ids)                 → PDFs
  Package: convert_rough(run_id)                       → rough markdown (all)
Agent reads rough markdown, evaluates full paper content
Agent picks ~10-15 papers (Tier 2 → Tier 3 transition)

──── TIER 3: Fine markdown + deep analysis ────
  Package: convert_fine(run_id, paper_ids)             → fine markdown (selected)
  Package: extract(run_id)                             → chunks from fine markdown
Agent synthesizes findings from fine markdown
```

The agent calls individual MCP tools and makes all selection/judgment
decisions. The package never autonomously discards papers — it computes
scores, fetches data, and returns results for the agent to interpret.

## New MCP Tools (Summary)

| Tool | Phase | Input | Output |
|------|-------|-------|--------|
| `compute_semantic_scores` | 3.5 | `run_id, topic` | `[{arxiv_id, semantic_score}]` |
| `enrich_candidates` | 2.7 | `run_id` | enriched count |
| `fetch_citations` | 4.5 | `paper_id, direction, limit` | `[CandidateRecord]` |
| `batch_fetch_citations` | 4.5 | `paper_ids, direction, limit` | `[CandidateRecord]` (deduped) |
| `compute_quality_scores` | 5.6 | `run_id, paper_ids?` | `[QualityScore]` |
| `get_venue_tier` | 5.6 | `venue_name` | `{tier, score}` |
| `convert_rough` | 7.6 | `run_id` | converted/skipped/failed counts |
| `convert_fine` | 7.6 | `run_id, paper_ids` | converted/skipped/failed counts |

---

## Files to Create (Summary)

```
src/research_pipeline/
  infra/
    rate_limit.py              (Phase 1.1 — generic rate limiter)
    retry.py                   (Phase 1.2 — retry decorator)
  models/
    quality.py                 (Phase 1.4 — QualityScore model)
  sources/
    semantic_scholar_source.py (Phase 2.1)
    openalex_source.py         (Phase 2.2)
    dblp_source.py             (Phase 2.3)
    enrichment.py              (Phase 2.5)
    citation_graph.py          (Phase 4.1)
  screening/
    embedding.py               (Phase 3.1–3.2)
  quality/
    __init__.py                (Phase 5)
    citation_metrics.py        (Phase 5.1)
    venue_scoring.py           (Phase 5.2)
    author_metrics.py          (Phase 5.3)
    composite.py               (Phase 5.4)
    data/
      core_rankings.json       (Phase 5.2 — static venue tiers)
  cli/
    cmd_expand.py              (Phase 4.4)
    cmd_quality.py             (Phase 5.5)
    cmd_convert_rough.py       (Phase 7.1 — rough conversion command)
    cmd_convert_fine.py        (Phase 7.2 — fine conversion command)
    cmd_index.py               (Phase 8.4)
  storage/
    global_index.py            (Phase 8.1)

tests/unit/
  test_rate_limit.py           (Phase 1.1)
  test_retry.py                (Phase 1.2)
  test_quality_model.py        (Phase 1.4)
  test_semantic_scholar.py     (Phase 2.1)
  test_openalex.py             (Phase 2.2)
  test_dblp.py                 (Phase 2.3)
  test_enrichment.py           (Phase 2.5)
  test_embedding.py            (Phase 3)
  test_citation_graph.py       (Phase 4)
  test_citation_metrics.py     (Phase 5.1)
  test_venue_scoring.py        (Phase 5.2)
  test_author_metrics.py       (Phase 5.3)
  test_composite_score.py      (Phase 5.4)
  test_convert_rough.py        (Phase 7.1)
  test_convert_fine.py         (Phase 7.2)
  test_global_index.py         (Phase 8)
```

## Files to Modify (Summary)

```
src/research_pipeline/
  models/candidate.py          (Phase 1.3 — add optional fields)
  models/conversion.py         (Phase 7.3 — add tier field)
  models/screening.py          (Phase 3.3 — add semantic_score)
  config/models.py             (Phases 2.4, 3.4, 4.3, 5.7, 7.8, 8.5)
  config/defaults.py           (same)
  config/loader.py             (env var overrides for new config)
  sources/base.py              (Phase 2.6 — enhanced dedup)
  screening/heuristic.py       (Phase 3.3 — blend semantic scores)
  cli/app.py                   (register expand, quality, convert-rough,
                                convert-fine, index commands)
  cli/cmd_search.py            (Phase 2 — add new sources)
  cli/cmd_extract.py           (Phase 7.4 — read fine→rough fallback)
  pipeline/orchestrator.py     (Phases 4, 5 — add expand+quality stages)
  storage/manifests.py         (Phase 6.2 — retry tracking)
  arxiv/rate_limit.py          (Phase 1.1 — refactor to use generic)
config.example.toml            (all phases — document new options)
pyproject.toml                 (Phases 2, 3 — new optional deps)
mcp_server/tools.py            (Phases 2.7, 3.5, 4.5, 5.6, 7.6 — new MCP tools)
mcp_server/schemas.py          (Phases 2.7, 3.5, 4.5, 5.6, 7.6 — new schemas)
```
