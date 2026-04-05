# Copilot Instructions — research-pipeline

## Summary

**research-pipeline** is a deterministic, stage-based Python 3.12+ pipeline for
searching, screening, downloading, converting, and summarizing academic papers
from arXiv and Google Scholar. It provides a Typer CLI (`research-pipeline`) and
an MCP server (10 tools via stdio). The codebase is ~4,500 lines of Python
across 40+ modules. Version is defined in `src/research_pipeline/__init__.py`
(currently `0.1.0`).

## Environment and bootstrap

This is a **uv-managed** project. Always prefix Python commands with `uv run`.
Never use bare `python`, `pytest`, or `pip`.

```bash
# Bootstrap (install all deps including dev, docling, and scholar extras)
uv sync --extra dev --extra docling --extra scholar

# Verify install works
uv run python -c "import research_pipeline; print(research_pipeline.__version__)"
# Expected output: 0.1.0
```

The project requires Python 3.12+. The `.python-version` file pins `3.12`. The
virtual environment lives at `.venv/` and is managed by uv.

## Build, test, lint, type-check (validated commands)

Run these in order. All commands have been validated and produce clean output on
the current codebase.

```bash
# 1. Run unit tests (196 tests, ~0.8s, no network required)
uv run pytest tests/unit/ -xvs

# 2. Format imports (isort, black-compatible profile)
uv run isort .

# 3. Format code (black, 88-char line length)
uv run black .

# 4. Lint (ruff, rules: E F I W UP B SIM, B008 ignored)
uv run ruff check . --fix

# 5. Type check (mypy strict mode — has existing errors on master, not blocking)
uv run mypy src/

# 6. Run all pre-commit hooks at once (combines steps 2-5 plus YAML/TOML checks)
uv run pre-commit run --all-files
```

**Important sequencing**: always run `isort` before `black` (isort may reorder
imports that black then re-wraps). Always run tests before committing.

The pre-commit config (`.pre-commit-config.yaml`) runs: trailing-whitespace,
end-of-file-fixer, check-yaml, check-toml, check-json, check-added-large-files
(max 1000KB), check-merge-conflict, detect-private-key, debug-statements,
check-ast, name-tests-test, check-docstring-first, isort, black, ruff, toml-sort.

There are no GitHub Actions workflows configured yet.

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
    cmd_extract.py              # extract stage
    cmd_summarize.py            # summarize stage
    cmd_run.py                  # end-to-end orchestration
    cmd_inspect.py              # run inspection
  models/                       # Pydantic domain models
    candidate.py                # CandidateRecord (arXiv paper metadata)
    query_plan.py               # QueryPlan (search terms & variants)
    screening.py                # CheapScoreBreakdown, RelevanceDecision
    download.py                 # DownloadManifestEntry
    conversion.py               # ConvertManifestEntry
    extraction.py               # ChunkMetadata, MarkdownExtraction
    summary.py                  # PaperSummary, SynthesisReport
    manifest.py                 # RunManifest, StageRecord, ArtifactRecord
  config/
    defaults.py                 # All default settings
    models.py                   # Pydantic config schemas
    loader.py                   # TOML loader + env var overrides
  arxiv/                        # arXiv API client, XML parser, dedup
  sources/                      # Multi-source adapter (arXiv, Scholar)
    base.py                     # SearchSource protocol
    arxiv_source.py             # arXiv adapter
    scholar_source.py           # Google Scholar (free scholarly + SerpAPI)
  screening/                    # BM25 heuristic scoring, LLM judge stub
  download/                     # Rate-limited PDF downloader
  conversion/                   # PDF→Markdown backends (Docling)
  extraction/                   # Markdown chunking & BM25 retrieval
  summarization/                # Per-paper + cross-paper synthesis
  pipeline/                     # Orchestrator & stage sequencing
  storage/                      # Workspace dirs, manifests, artifact hashing
  infra/                        # Cache, HTTP, logging, hashing, clock, paths
  llm/                          # LLM provider interface (experimental)

mcp_server/                     # FastMCP server (separate top-level package)
  server.py                     # Server entry, tool registration
  tools.py                      # Tool implementations (thin CLI adapters)
  schemas.py                    # Pydantic input/output schemas

tests/
  unit/                         # 196 fast unit tests (no network)
    test_<module>.py            # Each test file maps to a source module
  integration_offline/          # VCR-cassette offline integration tests
  live/                         # Tests with @pytest.mark.live (real arXiv API)
  fixtures/                     # Test data dirs (atom/, pdf/, markdown/, llm/)
```

## Architecture

7 sequential stages: **plan → search → screen → download → convert → extract →
summarize**. Each stage is idempotent and can be re-run independently.

- **Pipeline state**: tracked via `run_manifest.json` with SHA-256 artifact hashing
- **Configuration**: TOML file (`config.toml`) + environment variable overrides
- **Rate limiting**: arXiv polite mode (3s floor, 5s default) — global, thread-safe
- **CLI entry point**: registered in `pyproject.toml` as `research-pipeline`
- **MCP server**: `python -m mcp_server` (10 tools wrapping CLI logic)

## Code conventions

- Type hints on all function signatures (mypy strict mode)
- Google-style docstrings
- `logging` module only — never `print()` for operational output
- `pathlib.Path` over `os.path`
- Pydantic `BaseModel` for all domain objects
- Exception variable: `as exc`, not `as e`
- black formatting (88-char lines), isort (black profile)

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

## What not to do

- Do not track files in `runs/` or `workspace/` — they are gitignored outputs
- Do not commit `config.toml` — it may contain API keys; use `config.example.toml`
- Do not bypass arXiv rate limiting
- Do not use bare `python` or `pytest` — always `uv run`
- Do not add large files (pre-commit blocks files >1000KB)

Trust these instructions. Only search the codebase if the information here is
incomplete or found to be incorrect.
