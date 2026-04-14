# Troubleshooting & Configuration

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `command not found: research-pipeline` | Not installed | `pipx install research-pipeline` |
| No candidates found | Query too specific or window too narrow | Broaden terms, use `--source all`, expand to 12 months |
| Important paper missed | Synonym-blind queries | Add query variants with different vocabulary; check HuggingFace daily papers |
| Shortlist mostly irrelevant | Broad `must_terms` in BM25 | Cap must_terms to 2; use paper-screener for intelligent re-screening |
| Docling not installed | Missing docling extra | `pipx inject research-pipeline docling` |
| Marker not installed | Missing marker extra | `pipx inject research-pipeline marker-pdf` |
| PyMuPDF4LLM not installed | Missing pymupdf4llm extra | `pipx inject research-pipeline pymupdf4llm` |
| scholarly not installed | Missing scholarly package | `pipx inject research-pipeline scholarly` |
| Scholar SKIPPED on `--source all` | scholarly not injected into pipx venv | `pipx inject research-pipeline scholarly` (v0.3.1+ shows clear message) |
| Rate limit 429 | Too many API calls | Pipeline retries automatically with exponential backoff |
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

## Rate Limits

| Source | Limit | Notes |
|--------|-------|-------|
| arXiv | 1 req / 3s, single connection | Never parallel. 429 triggers exponential backoff |
| Google Scholar (free) | 10s+ between requests | May get captchas under heavy use |
| SerpAPI | 5s between requests | Paid, more reliable |
| HuggingFace | 0.5s between requests | Keyword-filtered daily papers feed |
| Semantic Scholar | 1s between requests | Used by `expand` and `quality` commands (not `--source`) |

## Search Sources

`--source all` searches **arXiv + Google Scholar + HuggingFace daily papers**
in parallel. Results are deduplicated by arXiv ID and normalized title.

Available source values for `--source`:
- `arxiv` — arXiv API (default)
- `scholar` — Google Scholar (requires scholarly or SerpAPI)
- `huggingface` — HuggingFace daily papers (keyword-filtered, recent papers)
- `all` — arXiv + Google Scholar + HuggingFace

Semantic Scholar, OpenAlex, and DBLP are used by the `expand` (citation graph)
and `quality` (author h-index) commands, but are **not** searchable via `--source`.

## Source Configuration

### Environment Variables
```bash
export RESEARCH_PIPELINE_SERPAPI_KEY=your-key       # Google Scholar paid API
export RESEARCH_PIPELINE_S2_API_KEY=your-s2-key     # Semantic Scholar (higher rate limits)
```

### config.toml
```toml
[sources]
default_sources = ["arxiv"]     # Searchable: arxiv, scholar, huggingface
semantic_scholar_api_key = ""   # Used by expand and quality commands
semantic_scholar_min_interval = 1.0

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
- **Time window**: Default 6 months. Expand to 12 only if sparse results.
- **Evidence-based**: Every summary claim must cite source (paper_id, section).
- CLI and MCP server share the same cache directory.

## Converter License Awareness

When choosing PDF-to-Markdown backends, be aware of license implications:

| Backend | License | Implications |
|---------|---------|-------------|
| Docling | MIT | Safe for all environments, including proprietary/commercial |
| Marker | GPL-3.0 | Copyleft — derivative works must also be GPL. May be unsuitable for proprietary pipelines |
| PyMuPDF4LLM | AGPL-3.0 | Network copyleft — if exposed as a service, source must be disclosed. Strictest license |
| Cloud backends (Mathpix, Datalab, etc.) | Proprietary/SaaS | Check vendor terms for data retention and usage rights |

**Policy recommendations**:
- For **enterprise or proprietary** environments: prefer `docling` or cloud backends
- For **open-source** projects: any local backend is suitable
- For **SaaS deployments** (e.g., behind an API): avoid AGPL backends unless source is disclosed
- Always document which backend was used in the run metadata for audit purposes
- When in doubt, configure `conversion.fallback_backends` to prefer MIT-licensed backends first

## MCP Server

Run with: `python -m mcp_server` (or `uv run python -m mcp_server`)

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
