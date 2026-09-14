# Troubleshooting & Configuration

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `command not found: research-pipeline` | Not installed | `pipx install research-pipeline` |
| No candidates found | Empty result or acquisition failure | Inspect source_coverage.json; broaden only after successful empty searches |
| Important paper missed | Synonym-blind queries | Add query variants with different vocabulary; check HuggingFace daily papers |
| Shortlist mostly irrelevant | Broad `must_terms` in BM25 | Cap must_terms to 2; use paper-screener for intelligent re-screening |
| Docling not installed | Missing docling extra | `pipx inject research-pipeline docling` |
| Marker not installed | Missing marker extra | `pipx inject research-pipeline marker-pdf` |
| PyMuPDF4LLM not installed | Missing pymupdf4llm extra | `pipx inject research-pipeline pymupdf4llm` |
| scholarly not installed | Missing scholarly package | `pipx inject research-pipeline scholarly` |
| Scholar SKIPPED on `--source all` | scholarly not injected into pipx venv | `pipx inject research-pipeline scholarly` (v0.3.1+ shows clear message) |
| Rate limit 429 | Server rate limit; other clients or shared IPs may contribute | Stop this source and respect the persisted cooldown and Retry-After |
| SerpAPI key not set | Using serpapi without key | Set `RESEARCH_PIPELINE_SERPAPI_KEY` env var |
| Quality scoring slow | Author h-index lookup via S2 API | Results are cached; subsequent runs faster |
| SPECTER2 model download | First use downloads ~440MB model | Cached after first download |
| `expand` requires paper-ids | Missing required argument | Add `--paper-ids "2401.12345,2402.67890"` |
| `convert-fine` requires paper-ids | Missing required argument | Add `--paper-ids "2401.12345"` |
| `converter_version: unknown` | Outdated pipeline (< 0.3.1) | Upgrade: version detection now uses `importlib.metadata` |
| Empty `logs/` directory | Outdated pipeline (< 0.3.1) | Upgrade: file logging now auto-enabled per run via `init_run()` |
| Double `pdf/pdf/` path in manifest | Outdated pipeline (< 0.3.1) | Upgrade: download path bug fixed in v0.3.1 |

## Bugs Fixed in v0.3.1

These bugs existed in v0.3.0 and are resolved in v0.3.1:

1. **Double `pdf/pdf/` path**: Download stage created `download/pdf/pdf/` instead of
   `download/pdf/` due to redundant path suffix in `cmd_download.py`.
2. **`write_jsonl` args reversed**: `convert-rough` and `convert-fine` commands
   passed arguments to `write_jsonl()` in the wrong order (would crash at runtime).
3. **`converter_version: unknown`**: Docling, Marker, and PyMuPDF4LLM backends used
   `getattr(pkg, "__version__")` which often fails; now uses `importlib.metadata.version()`.
4. **Empty logs directory**: No file logging was configured despite `logs/` directory
   being created. Now `init_run()` auto-attaches a JSONL file handler.
5. **Scholar fails silently**: When `--source all` was used without scholarly installed,
   the error message was generic. Now shows specific install instructions.

## Request Pacing

All source interval defaults are 30 seconds. This is a conservative application
default, not a claim about provider quotas or a guarantee of unblocking.
Existing explicit config overrides remain in effect; update them intentionally.

Default HTTP source clients share one budget per provider across processes under
~/.cache/research-pipeline/request-budgets. The lock serializes in-flight
requests. HTTP 429 ends acquisition for that source and persists a cooldown:
at least the source interval and the server's Retry-After value, or 15 minutes
when no usable Retry-After is provided. HTTP-date and numeric values work.
No retry is scheduled after the last attempt; deterministic 400 errors stop.

RESEARCH_PIPELINE_REQUEST_STATE_DIR can select a persistent shared directory.
Do not rotate or clear it to bypass a cooldown. Separate hosts/IP-sharing clients
need coordination beyond this directory. Custom injected HTTP sessions
remain the caller's responsibility. Scholar SDK operations use the same
coordination, but SDK-internal requests are not individually observable.

Inspect credential-safe source_http events for dispatch times, status,
content type, duration, and Retry-After. source_sdk events count SDK operations,
not wire requests. source_coverage.json records application query attempts,
date windows, source outcomes, and cooldown deadlines. Keep these levels distinct.

## Search Sources

`--source all` searches **arXiv + Google Scholar + Semantic Scholar +
OpenAlex + DBLP + HuggingFace daily papers** in parallel. Results are
deduplicated by arXiv ID, DOI, and normalized title.

Available source values for `--source`:
- `arxiv` — arXiv API (default)
- `scholar` — Google Scholar (requires scholarly or SerpAPI)
- `semantic_scholar` — Semantic Scholar
- `openalex` — OpenAlex
- `dblp` — DBLP
- `huggingface` — HuggingFace daily papers (keyword-filtered, recent papers)
- `all` — arXiv + Google Scholar + Semantic Scholar + OpenAlex + DBLP + HuggingFace

## Source Configuration

### Environment Variables
```bash
export RESEARCH_PIPELINE_SERPAPI_KEY=your-key       # Google Scholar paid API
export RESEARCH_PIPELINE_S2_API_KEY=your-s2-key     # Semantic Scholar (higher rate limits)
```

### config.toml
```toml
[sources]
enabled = ["arxiv"]             # Searchable: arxiv, scholar, semantic_scholar, openalex, dblp, huggingface
scholar_backend = "scholarly"   # scholarly or serpapi
serpapi_key = ""
semantic_scholar_api_key = ""
semantic_scholar_min_interval = 30.0
openalex_api_key = ""
openalex_min_interval = 30.0
dblp_min_interval = 30.0
huggingface_limit = 100

[screen]
use_semantic_reranking = false
embedding_model = "allenai/specter2"
embedding_batch_size = 32

[quality]
enabled = false

# Quality composite score formula:
# Q = w_c × Citation + w_v × Venue + w_a × Author + w_r × Recency
citation_weight = 0.35
venue_weight = 0.25
author_weight = 0.25
recency_weight = 0.15

[incremental]
enabled = false
global_index_path = ""
reuse_artifacts = true
```

## Caching

| Cache | Location | Retention |
|-------|----------|-----------|
| HTTP responses | `~/.cache/research-pipeline/` | 24 hours |
| Downloaded PDFs | `~/.cache/research-pipeline/pdf/` | 6 months |
| Converted Markdown | `~/.cache/research-pipeline/markdown/` | 6 months |

Before downloading a PDF, the pipeline checks the cache automatically.
If found, it copies to the run directory instead of re-downloading.

## Constraints

- **Query terms**: Cap AND-ed terms at 3. Prefer 2 `must_terms` for recall.
- **Synonym coverage**: ALWAYS generate variants with different vocabulary.
- **Time window**: Choose task-appropriate coverage. Automatic 6-to-12-month fallback applies only to successful sparse searches on date-aware sources.
- **Evidence-based**: Every summary claim must cite source (paper_id, section).
- CLI and MCP server share the same cache directory.

## MCP Server

Run with: `research-pipeline mcp serve` (or `uv run research-pipeline mcp serve`)

| MCP Tool | CLI Equivalent |
|----------|---------------|
| `tool_plan_topic` | `plan` |
| `tool_search` | `search` |
| `tool_screen_candidates` | `screen` |
| `tool_download_pdfs` | `download` |
| `tool_convert_pdfs` | `convert` (supports `backend` param) |
| `tool_extract_content` | `extract` |
| `tool_summarize_papers` | `summarize` |
| `tool_run_pipeline` | `run` |
| `tool_get_run_manifest` | `inspect` |
| `tool_convert_file` | `convert-file` (supports `backend` param) |
| `tool_list_backends` | — (list available converter backends) |
