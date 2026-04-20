# Copilot Instructions — research-pipeline

## Summary

**research-pipeline** is a deterministic, stage-based Python 3.12+ pipeline for
searching, screening, downloading, converting, and summarizing academic papers
from arXiv, Google Scholar, Semantic Scholar, OpenAlex, and DBLP. It provides a
Typer CLI (`research-pipeline`) and an MCP server (tools via stdio). The codebase
is ~48,000 lines of Python across 218 modules. Version is defined in
`src/research_pipeline/__init__.py` (currently `0.15.0`).

## Environment and bootstrap

This is a **uv-managed** project. Always prefix Python commands with `uv run`.
Never use bare `python`, `pytest`, or `pip`.

```bash
# Bootstrap (install all deps including dev, docling, and scholar extras)
uv sync --extra dev --extra docling --extra scholar

# Verify install works
uv run python -c "import research_pipeline; print(research_pipeline.__version__)"
# Expected output: 0.15.0
```

The project requires Python 3.12+. The `.python-version` file pins `3.12`. The
virtual environment lives at `.venv/` and is managed by uv.

## Build, test, lint, type-check (validated commands)

Run these in order. All commands have been validated and produce clean output on
the current codebase.

```bash
# 1. Run unit tests (3700+ tests, ~35s, no network required)
uv run pytest tests/unit/ -xvs

# 2. Format & lint (ruff handles formatting + linting, 88-char line length)
uv run ruff format .
uv run ruff check . --fix

# 3. Type check (mypy strict mode — enforced in CI, 0 errors)
uv run mypy src/

# 4. Run all pre-commit hooks at once (combines steps 2-3 plus YAML/TOML checks)
uv run pre-commit run --all-files
```

Always run tests before committing.

The pre-commit config (`.pre-commit-config.yaml`) runs: trailing-whitespace,
end-of-file-fixer, check-yaml, check-toml, check-json, check-added-large-files
(max 1000KB), check-merge-conflict, detect-private-key, debug-statements,
check-ast, name-tests-test, check-docstring-first, ruff-format, ruff-check,
toml-sort, bandit, validate-pyproject.

GitHub Actions CI runs: lint (pre-commit), test (3.12 + 3.13), typecheck (mypy),
security (pip-audit + pip-licenses).

## Project layout

```
pyproject.toml                  # All project config (deps, tools, scripts)
config.example.toml             # Pipeline config template (copy to config.toml)
.pre-commit-config.yaml         # Pre-commit hooks
uv.lock                         # Locked dependencies
.python-version                 # Python version pin (3.12)

src/research_pipeline/          # Main library (importable as research_pipeline)
  __init__.py                   # Package root, exports __version__
  cli/
    app.py                      # Typer root app, subcommand registration
    cmd_plan.py                 # plan stage implementation
    cmd_search.py               # search stage
    cmd_screen.py               # screen stage
    cmd_download.py             # download stage
    cmd_convert.py              # convert stage
    cmd_convert_file.py         # standalone PDF→Markdown (no workspace)
    cmd_convert_rough.py        # fast Tier 2 conversion (pymupdf4llm)
    cmd_convert_fine.py         # high-quality Tier 3 conversion
    cmd_extract.py              # extract stage
    cmd_summarize.py            # summarize stage
    cmd_expand.py               # citation graph expansion
    cmd_quality.py              # quality evaluation scoring
    cmd_run.py                  # end-to-end orchestration
    cmd_inspect.py              # run inspection
    cmd_index.py                # global paper index management
  models/                       # Pydantic domain models
    candidate.py                # CandidateRecord (multi-source paper metadata)
    query_plan.py               # QueryPlan (search terms & variants)
    screening.py                # CheapScoreBreakdown, RelevanceDecision
    download.py                 # DownloadManifestEntry (with retry tracking)
    conversion.py               # ConvertManifestEntry (with tier & retry)
    extraction.py               # ChunkMetadata, MarkdownExtraction
    summary.py                  # PaperSummary, SynthesisReport
    quality.py                  # QualityScore (composite evaluation)
    manifest.py                 # RunManifest, StageRecord, ArtifactRecord
  config/
    defaults.py                 # All default settings
    models.py                   # Pydantic config schemas
    loader.py                   # TOML loader + env var overrides
  arxiv/                        # arXiv API client, XML parser, dedup
  sources/                      # Multi-source adapter
    base.py                     # SearchSource protocol, cross-source dedup
    arxiv_source.py             # arXiv adapter
    scholar_source.py           # Google Scholar (free scholarly + SerpAPI)
    semantic_scholar_source.py  # Semantic Scholar API
    openalex_source.py          # OpenAlex API
    dblp_source.py              # DBLP API
    citation_graph.py           # Citation graph client (S2 API)
    enrichment.py               # Cross-source abstract enrichment
  screening/                    # BM25 heuristic scoring, LLM judge stub
    heuristic.py                # BM25 scoring with blended weights
    embedding.py                # SPECTER2 semantic re-ranking
  download/                     # Rate-limited PDF downloader with retry
  conversion/                   # PDF→Markdown backends (3 local + 5 cloud + fallback)
  memory/                        # Multi-tier memory system
    manager.py                  # MemoryManager (working + episodic + semantic)
    working.py                  # Bounded working memory (deque-based)
    cma_audit.py                # CMA six-property completeness auditor
    associative.py              # A-MEM associative linking (Jaccard + BFS)
    paging.py                   # MemGPT-style tiered paging + fault counters
  quality/                      # Quality evaluation
    citation_metrics.py         # Citation impact scoring
    venue_scoring.py            # Venue reputation (CORE rankings)
    author_metrics.py           # Author h-index credibility
    composite.py                # Weighted composite score
    graduated_rubric.py         # Graduated rubric + configurable criteria
    data/core_rankings.json     # Bundled CORE venue rankings
  extraction/                   # Markdown chunking & BM25 retrieval
  summarization/                # Per-paper + cross-paper synthesis
  pipeline/                     # Orchestrator & stage sequencing
    plan_revision.py            # TER plan-revision scoring + plateau detection
    topology.py                 # Pipeline profiles + adaptive difficulty routing
  storage/                      # Workspace dirs, manifests, artifact hashing
    workspace.py                # Stage directory management
    global_index.py             # SQLite global paper index (incremental)
  infra/                        # Cache, HTTP, logging, hashing, clock, paths
    rate_limit.py               # Generic thread-safe rate limiter
    retry.py                    # @retry decorator (backoff, jitter, Retry-After)
    entropy_monitor.py          # Shannon entropy rolling-window monitor
    failure_taxonomy.py         # Failure categories + UltraHorizon long-horizon modes
  security/                     # Security & adversarial robustness
    mcp_guard.py                # MCP tool registry, pinning, capability policies
    adversarial.py              # ToolTweak adversarial perturbation catalog
    trilemma.py                 # Defense-trilemma K^n budget monitor
  llm/                          # LLM provider interface (experimental)

mcp_server/                     # FastMCP server (separate top-level package)
    server.py                   # Server entry point, tool/resource/prompt registration
    tools.py                    # 16 pipeline tool implementations (thin CLI adapters)
    schemas.py                  # Pydantic input/output schemas
    resources.py                # 12 resource handlers (URI template based)
    prompts.py                  # 5 prompt templates for research workflows
    completions.py              # Auto-complete handler
    workflow/                   # Harness-engineered research workflow
      state.py                  # WorkflowState, stage transitions, persistence
      telemetry.py              # Three-surface logging (cognitive/operational/contextual)
      verification.py           # Structural output verification per stage
      context.py                # Token budgets, 5-stage paper compaction (ACC)
      monitoring.py             # Doom-loop detection, iteration drift tracking
      research.py               # Main orchestrator (~1100 lines)

.github/skills/research-pipeline/  # Bundled AI skill for Claude Code / Copilot
    SKILL.md                    # Skill definition and workflow instructions
    config.toml                 # Pipeline configuration template
    references/                 # Reference docs (sub-agents, troubleshooting, etc.)

tests/
  unit/                         # 3700+ fast unit tests (no network)
    test_<module>.py            # Each test file maps to a source module
  integration_offline/          # VCR-cassette offline integration tests
  live/                         # Tests with @pytest.mark.live (real arXiv API)
  fixtures/                     # Test data dirs (atom/, pdf/, markdown/, llm/)
```

## Architecture

7 core sequential stages: **plan → search → screen → download → convert →
extract → summarize**. Plus 5 auxiliary commands: **expand**, **quality**,
**convert-rough**, **convert-fine**, **index**. Each stage is idempotent and
can be re-run independently.

- **Pipeline state**: tracked via `run_manifest.json` with SHA-256 artifact hashing
- **Configuration**: TOML file (`config.toml`) + environment variable overrides
- **Rate limiting**: generic `RateLimiter` (`infra/rate_limit.py`), arXiv polite
  mode (3s floor, 5s default). Each source has its own limiter.
- **Retry**: `@retry` decorator (`infra/retry.py`) — exponential backoff, jitter,
  Retry-After header support
- **CLI entry point**: registered in `pyproject.toml` as `research-pipeline`
- **MCP server**: `python -m mcp_server` — 51 tools (with annotations and
  progress reporting), 15 resources (URI templates for run artifacts and
  workflow state), 6 prompts (research workflow templates), auto-completions,
  harness-engineered research workflow with sampling, elicitation, and
  6-layer harness (telemetry, context, governance, verification, monitoring,
  recovery)
- **Multi-source search**: arXiv + Google Scholar + Semantic Scholar + OpenAlex +
  DBLP with cross-source dedup (arXiv ID, DOI, normalized title)
- **Semantic re-ranking**: optional SPECTER2 embeddings in screen stage
- **Quality evaluation**: citation impact, venue reputation (CORE), author
  h-index, recency — weighted composite score
- **Two-tier conversion**: fast `convert-rough` (pymupdf4llm) + high-quality
  `convert-fine` (configured backend)
- **Incremental runs**: SQLite global paper index for cross-run dedup
- **Multi-account rotation**: online conversion backends support multiple
  accounts per service via `[[conversion.<backend>.accounts]]` TOML config.
  When a quota/rate limit is hit, the pipeline rotates to the next account.
- **Cross-service fallback**: `FallbackConverter` (`conversion/fallback.py`)
  wraps multiple backends and tries them in order. Configure via
  `conversion.fallback_backends` — an ordered list of backup backend names.

## Code conventions

- Type hints on all function signatures (mypy strict mode)
- Google-style docstrings
- `logging` module only — never `print()` for operational output
- `pathlib.Path` over `os.path`
- Pydantic `BaseModel` for all domain objects
- Exception variable: `as exc`, not `as e`
- ruff formatting (88-char lines) and import sorting

## Testing rules

- **Never modify existing tests** without explicit approval
- Write tests first when adding features (TDD)
- New unit tests go in `tests/unit/test_<module>.py`
- Mark live tests with `@pytest.mark.live`
- Use VCR cassettes for HTTP mocking
- Run specific tests, not the full suite: `uv run pytest tests/unit/test_foo.py::test_bar -xvs`

## Adding a new pipeline stage

1. Create domain model in `src/research_pipeline/models/`
2. Implement logic under `src/research_pipeline/<stage>/`
3. Add CLI command in `src/research_pipeline/cli/cmd_<stage>.py`
4. Register in `src/research_pipeline/cli/app.py`
5. Add stage to orchestrator in `src/research_pipeline/pipeline/`
6. Add MCP tool in `mcp_server/tools.py` and schema in `mcp_server/schemas.py`
7. Write unit tests in `tests/unit/test_<stage>.py`
8. Run format, lint, and tests before committing

## Adding a new converter backend

1. Create `src/research_pipeline/conversion/<name>_backend.py`
2. Subclass `ConverterBackend` from `conversion/base.py`
3. Decorate with `@register_backend("name")` from `conversion/registry.py`
4. Add config model (`*Account` + `*Config`) in `config/models.py`
5. Add account dispatch in `cli/cmd_convert.py` `_backend_kwargs_list()`
6. Add optional dependency in `pyproject.toml` extras
7. Write unit tests

## What not to do

- Do not track files in `runs/` or `workspace/` — they are gitignored outputs
- Do not commit `config.toml` — it may contain API keys; use `config.example.toml`
- Do not bypass arXiv rate limiting
- Do not use bare `python` or `pytest` — always `uv run`
- Do not add large files (pre-commit blocks files >1000KB)

Trust these instructions. Only search the codebase if the information here is
incomplete or found to be incorrect.
