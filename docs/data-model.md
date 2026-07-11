# Data Model: research-pipeline

## 1. Document Purpose

This document describes all data structures in `research-pipeline`:
Pydantic domain models, the JSON/JSONL file formats written per pipeline
stage, and the SQLite database schemas used for persistent state.

---

## 2. Design Principles

- **Pydantic v2 `BaseModel`** for all domain objects — serialisation,
  validation, and type-safety come for free.
- **Immutability preferred** — most models are read-after-write; mutation is
  confined to manifest update paths.
- **Backward compatibility** — new fields are added with `default=None` or
  `default_factory` to avoid breaking existing workspace artefacts.
- **SHA-256 content hashing** — every artefact tracks its hash for
  idempotency and integrity verification.

---

## 3. Domain Models (`src/research_pipeline/models/`)

### 3.1 CandidateRecord — paper metadata

Source: `models/candidate.py`

| Field | Type | Description |
|-------|------|-------------|
| `arxiv_id` | `str` | Base arXiv ID, e.g. `"2501.12345"` |
| `version` | `str` | Version string, e.g. `"v2"` |
| `title` | `str` | Paper title |
| `authors` | `list[str]` | Author names |
| `published` | `datetime` | v1 submission time (UTC) |
| `updated` | `datetime` | This version's submission time (UTC) |
| `categories` | `list[str]` | All arXiv categories |
| `primary_category` | `str` | Primary arXiv category |
| `abstract` | `str` | Paper abstract |
| `abs_url` | `str` | URL to abstract page |
| `pdf_url` | `str` | URL to PDF |
| `source` | `str` | Source system (default: `"arxiv"`) |
| `doi` | `str \| None` | DOI |
| `semantic_scholar_id` | `str \| None` | Semantic Scholar paper ID |
| `openalex_id` | `str \| None` | OpenAlex work ID |
| `citation_count` | `int \| None` | Total citation count |
| `influential_citation_count` | `int \| None` | Semantic Scholar influential citations |
| `venue` | `str \| None` | Publication venue name |
| `year` | `int \| None` | Publication year |

**Written to**: `search/candidates.jsonl` (one record per line)

---

### 3.2 QueryPlan — structured search plan

Source: `models/query_plan.py`

| Field | Type | Description |
|-------|------|-------------|
| `topic` | `str` | Original topic string |
| `normalized_topic` | `str` | Cleaned/normalized topic |
| `primary_query` | `str` | Main search query |
| `query_variants` | `list[str]` | Alternative query formulations |
| `arxiv_categories` | `list[str]` | Suggested arXiv categories |
| `sparsity_thresholds` | `SparsityThresholds` | Score thresholds for screening |
| `profile` | `str` | Pipeline profile (quick/standard/deep/auto) |
| `metadata` | `dict` | Arbitrary metadata |

**Written to**: `plan/query_plan.json`

---

### 3.3 Screening models

Source: `models/screening.py`

| Class | Purpose |
|-------|---------|
| `CheapScoreBreakdown` | BM25 score components per paper |
| `EvidenceQuote` | Supporting quote from paper text |
| `LLMJudgment` | LLM relevance judgment with reasoning |
| `RelevanceDecision` | Final screening decision (accept/reject/borderline) |

**Written to**: `screen/screened.jsonl` (one `RelevanceDecision` per paper)

---

### 3.4 Download and Conversion models

Source: `models/download.py`, `models/conversion.py`

| Class | Key Fields |
|-------|-----------|
| `DownloadManifestEntry` | `arxiv_id`, `pdf_path`, `sha256`, `status`, `retry_count`, `last_error` |
| `ConvertManifestEntry` | `arxiv_id`, `markdown_path`, `backend`, `tier`, `sha256`, `status`, `retry_count` |

**Written to**: `download/download_manifest.json`, `convert/convert_manifest.json`

---

### 3.5 Extraction models

Source: `models/extraction.py`

| Class | Key Fields |
|-------|-----------|
| `ChunkMetadata` | `chunk_id`, `paper_id`, `section`, `text`, `start_char`, `end_char`, `embedding?` |
| `ExtractedClaim` | `text`, `evidence_type`, `confidence` |
| `MarkdownExtraction` | `paper_id`, `markdown_path`, `chunks: list[ChunkMetadata]`, `extracted_claims` |

**Written to**: `extract/<arxiv_id>.extract.json`

---

### 3.6 Summary models

Source: `models/summary.py`

Key classes:

| Class | Description |
|-------|-------------|
| `PaperSummary` | Per-paper summary with themes, contributions, evidence snippets |
| `SynthesisReport` | Cross-paper synthesis: findings, agreements, disagreements, gaps |
| `SynthesisFinding` | A single synthesis finding with evidence pointers |
| `ContradictionRecord` | Contradicting claims across papers |
| `SynthesisQuality` | Quality assessment of the synthesis |
| `ConfidenceLevel` | `StrEnum`: `high`, `medium`, `low`, `unknown` |

**Written to**: `summarize/<arxiv_id>.summary.json`, `summarize/synthesis_report.{json,md}`

---

### 3.7 Quality model

Source: `models/quality.py`

| Class | Key Fields |
|-------|-----------|
| `QualityScore` | `arxiv_id`, `citation_score`, `venue_score`, `author_score`, `recency_score`, `composite_score`, `venue_tier?` |

Weights: citation 35%, venue 25%, author 25%, recency 15%

---

### 3.8 Manifest models

Source: `models/manifest.py`

| Class | Key Fields |
|-------|-----------|
| `RunManifest` | `run_id`, `created_at`, `package_version`, `config_snapshot`, `topic_input`, `stages`, `artifacts`, `llm_calls` |
| `StageRecord` | `stage_name`, `status`, `started_at`, `ended_at`, `duration_ms`, `output_paths`, `warnings`, `errors` |
| `ArtifactRecord` | `artifact_id`, `artifact_type`, `path`, `sha256`, `producer`, `inputs`, `created_at` |
| `LLMCallRecord` | `call_id`, `provider`, `model`, `input_hash`, `output_hash`, `token_usage`, `called_at`, `duration_ms` |

**Written to**: `<run_root>/run_manifest.json`
**Schema version**: `"1"` (in `RunManifest.schema_version`)

---

### 3.9 Claim and Evidence models

Source: `models/claim.py`, `models/evidence.py`

| Class | Description |
|-------|-------------|
| `AtomicClaim` | Single factual claim extracted from paper text |
| `ClaimDecomposition` | All claims for one paper |
| `EvidenceClass` | `StrEnum`: `supported`, `unsupported`, `contradicted` |
| `EvidenceStatement` | A statement with its evidence pointers |
| `EvidenceAggregation` | Aggregated evidence stripped of rhetoric |
| `RhetoricType` | `StrEnum`: hedge, filler, opinion, promotion |
| `RhetoricSpan` | A span of text identified as rhetoric |

---

### 3.10 Gap and Snowball models

Source: `models/gap.py`, `models/snowball.py`

| Class | Description |
|-------|-------------|
| `GapRecord` | A research gap identified in synthesis |
| `EvidenceMap` | Maps gap records to supporting evidence |
| `SnowballResult` | Citation snowball expansion result |
| `SnowballBudget` | Budget constraints for snowball expansion |
| `StopReason` | `StrEnum`: `budget_exhausted`, `no_new_papers`, `max_rounds` |

---

## 4. Stage Output File Formats

Every pipeline run writes files under `<workspace>/<run_id>/`:

```
<run_id>/
├── run_manifest.json          # RunManifest — full run metadata
├── plan/
│   └── query_plan.json        # QueryPlan
├── search/
│   └── candidates.jsonl       # list[CandidateRecord], one per line
├── screen/
│   ├── shortlist.json         # list[CandidateRecord] (filtered)
│   └── screened.jsonl         # list[RelevanceDecision], one per line
├── download/
│   ├── download_manifest.json # list[DownloadManifestEntry]
│   └── pdf/
│       └── <arxiv_id>.pdf
├── convert/
│   ├── convert_manifest.json  # list[ConvertManifestEntry]
│   └── markdown/
│       └── <arxiv_id>.md
├── extract/
│   └── <arxiv_id>.extract.json  # MarkdownExtraction
├── summarize/
│   ├── <arxiv_id>.summary.json  # PaperSummary
│   ├── synthesis_report.json    # SynthesisReport
│   ├── synthesis_report.md      # Rendered Markdown synthesis
│   └── synthesis.json           # (alias for synthesis_report.json in some paths)
└── logs/
    ├── pipeline.jsonl           # Structured log output
    ├── traces.jsonl             # Eval log trace channel
    └── audit.db                 # AuditDB (SQLite — see §5.5)
```

---

## 5. SQLite Database Schemas

### 5.1 Global Paper Index

**File**: `~/.cache/research-pipeline/paper_index.db`
**Purpose**: Cross-run paper deduplication

```sql
CREATE TABLE papers (
    arxiv_id      TEXT,
    doi           TEXT,
    s2_id         TEXT,
    title         TEXT,
    abstract      TEXT DEFAULT '',
    run_id        TEXT NOT NULL,
    stage         TEXT NOT NULL,
    pdf_path      TEXT,
    markdown_path TEXT,
    summary_path  TEXT,
    pdf_sha256    TEXT,
    indexed_at    TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_id)
);
CREATE INDEX idx_doi ON papers(doi);
CREATE INDEX idx_s2_id ON papers(s2_id);
CREATE INDEX idx_title ON papers(title);

-- Full-text search
CREATE VIRTUAL TABLE papers_fts USING fts5(arxiv_id, title, abstract);
```

---

### 5.2 Episodic Memory

**File**: Configurable (default: `~/.cache/research-pipeline/episodic_memory.db`)
**Purpose**: Records of past pipeline runs for CBR and resume

```sql
CREATE TABLE episodes (
    run_id            TEXT PRIMARY KEY,
    topic             TEXT NOT NULL,
    profile           TEXT DEFAULT 'standard',
    started_at        TEXT NOT NULL,
    completed_at      TEXT,
    stages_completed  TEXT DEFAULT '[]',   -- JSON array
    paper_count       INTEGER DEFAULT 0,
    shortlist_count   INTEGER DEFAULT 0,
    synthesis_summary TEXT DEFAULT '',
    gaps_found        TEXT DEFAULT '[]',   -- JSON array
    key_decisions     TEXT DEFAULT '[]',   -- JSON array
    outcome           TEXT DEFAULT '',
    metadata          TEXT DEFAULT '{}'    -- JSON object
);
CREATE INDEX idx_topic ON episodes(topic);
CREATE INDEX idx_started_at ON episodes(started_at);
```

---

### 5.3 Knowledge Graph

**File**: Configurable (default: `~/.cache/research-pipeline/knowledge_graph.db`)
**Purpose**: Entity and triple store for research knowledge

```sql
CREATE TABLE entities (
    entity_id   TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name        TEXT NOT NULL,
    properties  TEXT DEFAULT '{}',     -- JSON object
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX idx_entity_type ON entities(entity_type);
CREATE INDEX idx_entity_name ON entities(name);

CREATE TABLE triples (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id       TEXT NOT NULL,
    relation         TEXT NOT NULL,
    object_id        TEXT NOT NULL,
    provenance_paper TEXT DEFAULT '',
    provenance_run   TEXT DEFAULT '',
    confidence       REAL DEFAULT 1.0,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES entities(entity_id),
    FOREIGN KEY (object_id) REFERENCES entities(entity_id),
    UNIQUE (subject_id, relation, object_id, provenance_paper)
);
CREATE INDEX idx_triple_subject ON triples(subject_id);
CREATE INDEX idx_triple_object ON triples(object_id);
CREATE INDEX idx_triple_relation ON triples(relation);
CREATE INDEX idx_triple_provenance ON triples(provenance_paper);
```

---

### 5.4 Case-Based Reasoning Store

**File**: `~/.cache/research-pipeline/cbr_cases.db`
**Purpose**: Past research strategies for retrieval and adaptation

```sql
CREATE TABLE cases (
    case_id            TEXT PRIMARY KEY,
    topic              TEXT NOT NULL,
    query_terms        TEXT DEFAULT '[]',   -- JSON array
    sources_used       TEXT DEFAULT '[]',   -- JSON array
    screening_config   TEXT DEFAULT '{}',   -- JSON object
    pipeline_profile   TEXT DEFAULT 'standard',
    paper_count        INTEGER DEFAULT 0,
    shortlist_count    INTEGER DEFAULT 0,
    synthesis_quality  REAL DEFAULT 0.0,
    pass_at_k          REAL DEFAULT 0.0,
    contamination_score REAL DEFAULT 0.0,
    outcome            TEXT DEFAULT 'unknown',
    strategy_notes     TEXT DEFAULT '',
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    metadata           TEXT DEFAULT '{}'    -- JSON object
);
CREATE INDEX idx_cases_topic ON cases(topic);
CREATE INDEX idx_cases_outcome ON cases(outcome);
CREATE INDEX idx_cases_quality ON cases(synthesis_quality);

CREATE TABLE case_adaptations (
    adaptation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_case_id TEXT NOT NULL,
    target_case_id TEXT NOT NULL,
    adaptation_type TEXT NOT NULL,
    changes_applied TEXT DEFAULT '{}',  -- JSON object
    quality_delta   REAL DEFAULT 0.0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_case_id) REFERENCES cases(case_id),
    FOREIGN KEY (target_case_id) REFERENCES cases(case_id)
);
```

---

### 5.5 Evaluation Audit Log

**File**: `<run_root>/logs/audit.db`
**Purpose**: Structured audit of every stage action (who/what/when)

```sql
CREATE TABLE audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
    run_id       TEXT NOT NULL DEFAULT '',
    stage        TEXT NOT NULL DEFAULT '',
    action       TEXT NOT NULL DEFAULT '',
    input_hash   TEXT NOT NULL DEFAULT '',
    output_hash  TEXT NOT NULL DEFAULT '',
    model        TEXT NOT NULL DEFAULT '',
    tokens_used  INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    details      TEXT NOT NULL DEFAULT '{}'  -- JSON object
);
CREATE INDEX idx_audit_stage ON audit_log(stage);
CREATE INDEX idx_audit_run ON audit_log(run_id);
```

---

### 5.6 Additional Databases

| Database | Default Path | Purpose |
|----------|-------------|---------|
| Dual metrics | `~/.cache/research-pipeline/dual_metrics.db` | Pass@k + Pass[k] benchmark results |
| Blinding audit | `<workspace>/.blinding_audits.db` | A/B blinding contamination audit (workspace-level) |
| Feedback store | `~/.cache/research-pipeline/feedback.db` | User paper accept/reject feedback |
| Briefing feedback | `<briefing-root>/feedback/feedback.db` | User briefing item feedback |
| Topic memory | `<workspace>/topic_memory.db` | Briefing topic novelty memory |
| Versioned entries | `~/.cache/research-pipeline/versioned_memory.db` | Versioned key-value memory store |

---

## 6. Configuration Model

The configuration is loaded from `config.toml` using `src/research_pipeline/config/loader.py`.
The schema is defined in `src/research_pipeline/config/models.py`.

### Top-level sections

| Section | Purpose |
|---------|---------|
| `[pipeline]` | Profile, max iterations, output directory |
| `[arxiv]` | arXiv rate limiting, max results, categories |
| `[cache]` | HTTP cache TTL, cache directory |
| `[sources]` | Enabled sources, per-source rate limits |
| `[screen]` | Shortlist size, BM25 weights, MMR diversity |
| `[download]` | Concurrency, timeout, retry limits |
| `[conversion]` | Backend selection, tier thresholds, per-backend credentials |
| `[llm]` | Provider, model, phase routing |
| `[quality]` | Score weights, CORE rankings override |
| `[search]` | Max results per source, dedup strategy |
| `[incremental]` | Global index path, dedup fields |
| `[briefing]` | Source registry, polling schedule, output directory |

Full reference: see `config.example.toml` and `docs/api-reference.md`.

---

## 7. MCP Schemas

MCP tool input/output schemas are defined in:
- `src/research_pipeline/mcp_server/schemas.py` — Pydantic input models
- `src/research_pipeline/mcp_server/tools.py` — return types

All MCP tools accept and return JSON-serialisable Pydantic models.
See `docs/api-reference.md` §12 for the full MCP tool catalogue.

---

## 8. Briefing Pipeline Models

The daily AI Intelligence briefing pipeline uses additional Pydantic models
defined in `src/research_pipeline/briefing/`:

| Class | Module | Description |
|-------|--------|-------------|
| `IntelligenceEvent` | `briefing/models.py` | A single raw intelligence event from any source |
| `BriefingCluster` | `briefing/rank.py` | A deduplicated + ranked cluster of related events |
| `SourceAdapterConfig` | `briefing/registry.py` | Configuration for a briefing source adapter |
| `DossierEntry` | `briefing/dossier/` | A hot-topic deep-dive entry |
| `WeeklySynthesis` | `briefing/weekly.py` | Weekly trend synthesis memo |

---

## 9. Version and Compatibility

- `RunManifest.schema_version = "1"` — identifies the manifest format version
- All new optional fields use `default=None` or `default_factory=list/dict`
- Old workspace artefacts from v0.15.x+ can be read by current code
- Config format is backward-compatible; new sections are optional
