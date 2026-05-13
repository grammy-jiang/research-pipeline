# Operations Runbook: research-pipeline

## 1. Document Purpose

This runbook covers installation, configuration, monitoring, incident response,
and routine maintenance for `research-pipeline`. It is aimed at operators and
system administrators managing the tool in production or shared-server environments.

---

## 2. Installation

### 2.1 Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.12 |
| pip / uv | any recent |
| Disk (base) | ~200 MB (plus run outputs and PDF downloads) |
| Disk (with docling) | ~2 GB (docling model files) |
| Memory | ≥ 2 GB RAM |
| Network | Required for search, download, cloud conversion |

### 2.2 Install from PyPI

```bash
# Minimal install (plan, search, screen, summarize with abstract-only data)
pip install research-pipeline

# With recommended extras (full pipeline)
pip install "research-pipeline[docling,scholar,reranker]"

# All extras
pip install "research-pipeline[docling,marker,scholar,serpapi,reranker,llm,dev]"
```

### 2.3 Verify installation

```bash
research-pipeline version
# Expected output: research-pipeline 0.17.14
```

### 2.4 Optional extras reference

| Extra | Installs | Required for |
|-------|----------|-------------|
| `docling` | docling ≥ 2.0 | High-quality PDF→Markdown (recommended) |
| `marker` | marker-pdf ≥ 1.10 | Alternative PDF converter (GPL-3.0 licensed) |
| `pymupdf4llm` | pymupdf4llm ≥ 0.0.17 | Fast Tier 2 conversion |
| `mineru` | magic-pdf ≥ 0.9 | MinerU PDF converter |
| `scholar` | scholarly ≥ 1.7 | Google Scholar (free) source |
| `serpapi` | google-search-results ≥ 2.4 | Google Scholar via SerpAPI |
| `reranker` | sentence-transformers ≥ 2.2 | SPECTER2 semantic re-ranking |
| `llm` | (optional LLM deps) | LLM-based summarization/analysis |
| `mathpix` | — | Mathpix cloud OCR (API key required) |
| `datalab` | datalab-python-sdk ≥ 0.1 | DataLab cloud conversion |
| `llamaparse` | llama-cloud ≥ 1.0 | LlamaParse cloud conversion |
| `mistral-ocr` | — | Mistral OCR API (API key required) |
| `openai-vision` | openai ≥ 1.0, PyMuPDF ≥ 1.24 | OpenAI Vision conversion |
| `dev` | pytest, ruff, mypy, etc. | Development and testing |

> **Note**: `marker` (GPL-3.0) and `pymupdf4llm` (AGPL-3.0 via PyMuPDF) have
> copyleft licenses. Do not include these in production deployments unless
> copyleft obligations are accepted. Use `docling` instead.

---

## 3. Configuration

### 3.1 Create config.toml

```bash
# Copy the annotated example
cp config.example.toml config.toml

# Or point to a config file elsewhere
export RESEARCH_PIPELINE_CONFIG=/etc/research-pipeline/config.toml
```

### 3.2 Configuration precedence

1. `RESEARCH_PIPELINE_CONFIG` environment variable (path to TOML file)
2. `config.toml` in the current working directory
3. Built-in defaults in `config/defaults.py`

### 3.3 Essential configuration sections

```toml
[pipeline]
workspace = "runs"        # Where run outputs are written
profile = "standard"      # quick | standard | deep | auto

[arxiv]
max_results = 50          # Papers fetched per query
rate_limit_seconds = 5.0  # Min seconds between requests (min: 3.0)

[screen]
shortlist_size = 20       # Papers that proceed to download

[conversion]
backend = "docling"       # Primary PDF→Markdown backend

[sources]
enabled = ["arxiv", "semantic_scholar", "openalex"]

[llm]
provider = "openai_compatible"
base_url = "http://localhost:11434/v1"   # Ollama example
model = "llama3.2"
```

### 3.4 API keys

All API keys belong in `config.toml` (gitignored) or environment variables.
**Never commit credentials to the repository.**

| Service | Config key | Environment variable |
|---------|-----------|---------------------|
| SerpAPI (Google Scholar) | `[sources.serpapi] key` | `SERPAPI_KEY` |
| Semantic Scholar | `[sources.semantic_scholar] api_key` | `S2_API_KEY` |
| Mathpix | `[conversion.mathpix] app_id` / `app_key` | `MATHPIX_APP_ID`, `MATHPIX_APP_KEY` |
| Mistral OCR | `[conversion.mistral_ocr] api_key` | `MISTRAL_API_KEY` |
| OpenAI (vision) | `[conversion.openai_vision] api_key` | `OPENAI_API_KEY` |
| Codecov | CI only | `CODECOV_TOKEN` (GitHub secret) |

---

## 4. Workspace and File Locations

### 4.1 Runtime workspace

All pipeline run outputs are written under the configured workspace directory
(default: `runs/` in the current working directory):

```
runs/
└── <run_id>/             # 12-char hex run ID
    ├── run_manifest.json
    ├── plan/
    │   └── query_plan.json
    ├── search/
    │   └── candidates.jsonl
    ├── screen/
    │   ├── shortlist.json
    │   └── screened.jsonl
    ├── download/
    │   ├── download_manifest.json
    │   └── pdf/
    ├── convert/
    │   ├── convert_manifest.json
    │   └── markdown/
    ├── extract/
    ├── summarize/
    └── logs/
        ├── pipeline.jsonl   # Structured log output
        ├── traces.jsonl     # Execution flow trace
        └── audit.db         # Audit SQLite database
```

> **Note**: The `runs/` and `workspace/` directories are gitignored.
> Do not commit run outputs.

### 4.2 User-level persistent data

| Path | Purpose |
|------|---------|
| `~/.cache/research-pipeline/paper_index.db` | Global paper dedup index |
| `~/.cache/research-pipeline/episodic_memory.db` | Past run episodes |
| `~/.cache/research-pipeline/knowledge_graph.db` | Knowledge graph |
| `~/.cache/research-pipeline/cbr_cases.db` | Case-based reasoning store |
| `~/.cache/research-pipeline/dual_metrics.db` | Pass@k benchmark results |
| `~/.cache/research-pipeline/blinding_audit.db` | Blinding audit results |
| `~/.cache/research-pipeline/http_cache/` | HTTP response cache |
| `~/.claude/skills/research-pipeline/` | Installed Claude skill |
| `~/.codex/skills/research-pipeline/` | Installed Copilot skill |
| `~/.claude/agents/` | Installed sub-agents |
| `~/.config/research-pipeline/mcp.json` | MCP config snippet |

---

## 5. Running the Pipeline

### 5.1 Basic end-to-end research run

```bash
# Standard run (all 7 stages)
research-pipeline run "transformer architectures for time series"

# Deep research (slower, more thorough)
research-pipeline run "LLM reasoning capabilities" --profile deep

# Quick abstract-only scan
research-pipeline run "diffusion models" --profile quick

# Resume a failed run
research-pipeline run "my topic" --run-id <run_id>
```

### 5.2 Check run status

```bash
research-pipeline inspect --run-id <run_id>
```

Output includes per-stage status, artifact list, and any errors.

### 5.3 Daily AI Intelligence briefing

```bash
# Full briefing pipeline
research-pipeline brief run

# With specific date
research-pipeline brief run --date 2025-07-01

# Check configured sources
research-pipeline brief list-sources

# Check source gateway connectivity
research-pipeline brief gateway-check
```

### 5.4 MCP server

```bash
# Start MCP server (stdio transport)
research-pipeline mcp serve

# In a shell that connects to an MCP host:
# Configure in the host's MCP client config pointing to this command
```

---

## 6. Monitoring

### 6.1 Log output

The pipeline writes structured logs to two locations:

```bash
# Real-time log (JSON lines)
tail -f runs/<run_id>/logs/pipeline.jsonl | python -m json.tool

# Execution traces
tail -f runs/<run_id>/logs/traces.jsonl
```

### 6.2 Three-channel eval logging

```bash
# Summary across all channels for a run
research-pipeline eval-log --run-id <run_id> --channel summary

# Execution traces (timing, stage entry/exit)
research-pipeline eval-log --run-id <run_id> --channel traces

# Audit log (who/what/when)
research-pipeline eval-log --run-id <run_id> --channel audit

# Filter by stage
research-pipeline eval-log --run-id <run_id> --channel traces --stage screen
```

### 6.3 Run manifest inspection

```bash
# Human-readable run status
research-pipeline inspect --run-id <run_id>

# The JSON manifest has complete stage timing and artifact hashes
cat runs/<run_id>/run_manifest.json | python -m json.tool
```

### 6.4 Global index status

```bash
# List all papers in the global index
research-pipeline index --list

# Garbage-collect stale entries
research-pipeline index --gc
```

### 6.5 Memory system stats

```bash
research-pipeline memory-stats
research-pipeline memory-episodes
research-pipeline memory-search "transformer"
```

---

## 7. Incident Response

### 7.1 Failure category taxonomy

The pipeline classifies failures into these categories:

| Category | Description | Typical fix |
|----------|-------------|------------|
| `retrieval_miss` | No papers found matching query | Broaden query, enable more sources |
| `conversion_error` | PDF→Markdown conversion failed | Try different backend, check PDF validity |
| `synthesis_gap` | Synthesis skipped or incomplete | Check LLM config, increase shortlist size |
| `screening_miss` | All papers screened out | Loosen BM25 thresholds or increase max_results |
| `download_failure` | PDF download failed | Check network; arXiv may be rate-limiting |
| `extraction_error` | Markdown extraction failed | Check Markdown quality from conversion step |
| `llm_error` | LLM call failed | Check LLM config; test with `curl` to LLM endpoint |
| `rate_limit` | API rate limit hit | Reduce `max_results`; increase `rate_limit_seconds` |
| `validation_error` | Schema validation failed | Report a bug; inspect the artifact in question |
| `timeout` | Stage exceeded time limit | Reduce scope; check network latency |
| `config_error` | Invalid configuration | Check `config.toml` against `config.example.toml` |

### 7.2 Run recovery

If a run fails partway through, resume it without re-running completed stages:

```bash
# Resume from last successful stage
research-pipeline run "original topic" --run-id <failed_run_id>

# Or re-run a specific stage
research-pipeline screen --run-id <run_id>
```

### 7.3 PDF download failures

```bash
# Check which PDFs failed to download
cat runs/<run_id>/download/download_manifest.json | python -c "
import json, sys
for r in json.load(sys.stdin):
    if r['status'] != 'success':
        print(r['arxiv_id'], r['status'], r.get('last_error',''))
"

# Re-run download (skips already-downloaded PDFs)
research-pipeline download --run-id <run_id>
```

### 7.4 Conversion failures

```bash
# Check conversion manifest for failures
cat runs/<run_id>/convert/convert_manifest.json | python -c "
import json, sys
for r in json.load(sys.stdin):
    if r['status'] != 'success':
        print(r['arxiv_id'], r['backend'], r.get('last_error',''))
"

# Re-run with a different backend
research-pipeline convert --run-id <run_id> --backend marker
```

### 7.5 LLM issues

```bash
# Test LLM connectivity
curl http://localhost:11434/v1/models  # Ollama

# Check phase routing
research-pipeline mcp serve &
# Then call: research-pipeline eval-log to see LLM call records
```

---

## 8. Maintenance

### 8.1 Disk cleanup

Run outputs accumulate quickly (each full run can be 50–500 MB including PDFs):

```bash
# List runs sorted by size
du -sh runs/*/ | sort -h

# Remove a specific run (verify before deleting)
rm -rf runs/<run_id>/

# Remove all runs (dangerous — confirm before executing)
# This requires human approval (HC3)
```

### 8.2 HTTP cache eviction

```bash
# HTTP responses are cached at ~/.cache/research-pipeline/http_cache/
# Default TTL: 24 hours (configurable in [cache] section)
# To clear:
rm -rf ~/.cache/research-pipeline/http_cache/
```

### 8.3 Global paper index maintenance

```bash
# Remove entries for runs that no longer exist on disk
research-pipeline index --gc

# View indexed paper count
research-pipeline index --list | tail -1
```

### 8.4 Dependency security audit

```bash
# Check for vulnerable packages
uv run pip-audit

# Check license compatibility
uv run pip-licenses --order=license --format=markdown
```

### 8.5 Updating

```bash
pip install --upgrade research-pipeline

# Or with uv:
uv add research-pipeline@latest
```

---

## 9. Multi-Account Rotation and Fallback

For high-volume deployments, configure multiple accounts for cloud conversion
backends and a fallback chain:

```toml
# Multiple Mathpix accounts — pipeline rotates when quota is hit
[[conversion.mathpix.accounts]]
app_id = "account1_id"
app_key = "account1_key"

[[conversion.mathpix.accounts]]
app_id = "account2_id"
app_key = "account2_key"

# Fallback chain: try docling first, then marker, then pymupdf4llm
[conversion]
fallback_backends = ["docling", "marker", "pymupdf4llm"]
```

---

## 10. Security Checklist

Before deploying in a shared or production environment:

- [ ] `config.toml` is gitignored and not world-readable (`chmod 600 config.toml`)
- [ ] API keys are not set as environment variables in shared shell profiles
- [ ] `detect-secrets` baseline is current: `detect-secrets scan > .secrets.baseline`
- [ ] Network egress is restricted to approved destinations (see HC5)
- [ ] `pip-audit` shows no critical vulnerabilities
- [ ] `pymupdf4llm` / `marker` extras are not installed unless AGPL/GPL obligations accepted
- [ ] MCP server is only accessible to trusted MCP clients

---

## 11. Appendix: Quick Command Reference

| Goal | Command |
|------|---------|
| Full research run | `research-pipeline run "topic"` |
| Deep research run | `research-pipeline run "topic" --profile deep` |
| Resume failed run | `research-pipeline run "topic" --run-id <id>` |
| Inspect run status | `research-pipeline inspect --run-id <id>` |
| Check logs | `research-pipeline eval-log --run-id <id> --channel summary` |
| Export HTML report | `research-pipeline export-html --run-id <id>` |
| Export BibTeX | `research-pipeline export-bibtex --run-id <id>` |
| Daily briefing | `research-pipeline brief run` |
| Convert single PDF | `research-pipeline convert-file paper.pdf` |
| Global index status | `research-pipeline index --list` |
| Memory stats | `research-pipeline memory-stats` |
| MCP server | `research-pipeline mcp serve` |
| Install AI skill | `research-pipeline setup` |
| Show version | `research-pipeline version` |
