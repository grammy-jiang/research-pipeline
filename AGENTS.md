# AGENTS.md

## Project overview

**research-pipeline** is a deterministic, stage-based Python pipeline for
searching, screening, downloading, converting, and summarizing academic papers
from arXiv, Google Scholar, Semantic Scholar, OpenAlex, and DBLP. It provides
both a Typer CLI and an MCP (Model Context Protocol) server.

| Key | Value |
|-----|-------|
| Language | Python 3.12+ |
| Build system | uv (`uv_build` backend) |
| Package name | `research-pipeline` (PyPI) / `research_pipeline` (import) |
| Version | defined in `src/research_pipeline/__init__.py` |
| License | MIT |
| Entry point | `research-pipeline` CLI (registered in `pyproject.toml`) |

## Repository layout

```
├── AGENTS.md                     # This file — agent instructions
├── CLAUDE.md                     # Claude Code entry (imports this file)
├── .github/
│   └── copilot-instructions.md   # GitHub Copilot instructions
├── pyproject.toml                # Project metadata, deps, tool config
├── config.example.toml           # Pipeline configuration template
├── .pre-commit-config.yaml       # Pre-commit hooks
├── uv.lock                       # Locked dependencies
├── src/research_pipeline/        # Main library
│   ├── cli/                      # Typer CLI commands
│   ├── models/                   # Pydantic domain models
│   ├── config/                   # Configuration loading & schemas
│   ├── arxiv/                    # arXiv API client, parser, dedup
│   ├── sources/                  # Multi-source adapter (arXiv, Scholar, S2, OpenAlex, DBLP)
│   ├── screening/                # Heuristic BM25 scoring, SPECTER2 embeddings, LLM judge
│   ├── download/                 # Rate-limited PDF downloader with retry
│   ├── conversion/               # PDF→Markdown backends (3 local + 5 cloud + fallback)
│   ├── quality/                  # Quality evaluation (citations, venue, author, composite)
│   ├── extraction/               # Markdown chunking & retrieval
│   ├── summarization/            # Per-paper + cross-paper synthesis
│   ├── pipeline/                 # Orchestrator & stage sequencing
│   ├── storage/                  # Workspace, manifests, artifacts, global index
│   ├── infra/                    # Cache, HTTP, logging, hashing, clock, rate limiting, retry
│   └── llm/                      # LLM provider interface (experimental)
├── mcp_server/                   # FastMCP server
│   ├── server.py                 # Server entry point
│   ├── tools.py                  # Tool implementations
│   └── schemas.py                # Input/output Pydantic schemas
├── tests/
│   ├── unit/                     # Fast, isolated unit tests
│   ├── integration_offline/      # VCR-cassette offline integration tests
│   ├── live/                     # Tests requiring live arXiv access
│   └── fixtures/                 # Test data (atom XML, PDFs, markdown)
├── docs/                         # Project documentation
├── runs/                         # Pipeline run outputs (gitignored)
└── workspace/                    # MCP workspace outputs (gitignored)
```

## Setup commands

```bash
# Install all dependencies including dev extras
uv sync --extra dev --extra docling --extra scholar

# Activate the virtual environment (if not using uv run)
source .venv/bin/activate
```

## Build and test

```bash
# Run all unit tests (fast, no network)
uv run pytest tests/unit/ -xvs

# Run a single test
uv run pytest tests/unit/test_models.py::test_candidate_roundtrip -xvs

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run live tests (requires network, against real arXiv API)
uv run pytest tests/live/ -xvs -m live
```

## Format and lint

Always run these before committing:

```bash
# Import sorting → code formatting → linting
uv run isort .
uv run black .
uv run ruff check . --fix

# Type checking (strict mode)
uv run mypy src/
```

Pre-commit hooks enforce this automatically:

```bash
uv run pre-commit run --all-files
```

## Code style

- **Formatter**: black (88 char line length)
- **Import sorting**: isort (black-compatible profile)
- **Linter**: ruff (select: E, F, I, W, UP, B, SIM)
- **Type checking**: mypy (strict mode, Python 3.12)
- **Docstrings**: Google style
- Use `logging` module, never `print()` for output
- Use `pathlib.Path`, not `os.path`
- Use f-strings for formatting
- Use type hints on all function signatures
- Use `as exc` for exception variables, not `as e`

## Testing conventions

- Framework: pytest
- **Never modify existing tests** without explicit approval
- Write tests first when adding new features (TDD)
- Run only the relevant test file/function, not the entire suite
- Place new unit tests in `tests/unit/test_<module>.py`
- Test fixture files go in `tests/fixtures/<type>/`
- Mark live-network tests with `@pytest.mark.live`
- Use VCR cassettes for HTTP-dependent tests (`tests/fixtures/http_cassettes/`)

## Architecture patterns

1. **Stage-based pipeline**: 7 core sequential stages —
   plan → search → screen → download → convert → extract → summarize
   Plus 5 auxiliary commands: expand, quality, convert-rough, convert-fine, index
2. **Each stage is idempotent** and can be re-run independently
3. **Pydantic models** for all domain objects (`src/research_pipeline/models/`)
4. **Configuration**: TOML config + env var overrides
   (`src/research_pipeline/config/`)
5. **Manifest tracking**: every run has `run_manifest.json` with full artifact
   lineage and SHA-256 hashes
6. **Rate limiting**: generic `RateLimiter` (`infra/rate_limit.py`) extended by
   `ArxivRateLimiter` (3s floor). Each source has its own limiter.
7. **Multi-source**: arXiv + Google Scholar + Semantic Scholar + OpenAlex + DBLP
   with cross-source dedup (by arXiv ID, DOI, and normalized title)
8. **Multi-account rotation**: online conversion backends support multiple
   accounts per service via `[[conversion.<backend>.accounts]]` TOML config.
   When a quota/rate limit is hit, the pipeline rotates to the next account.
9. **Cross-service fallback**: `FallbackConverter` (`conversion/fallback.py`)
   wraps multiple backends and tries them in order. Configure via
   `conversion.fallback_backends` — an ordered list of backup backend names.
10. **Retry & error recovery**: `@retry` decorator (`infra/retry.py`) with
    exponential backoff, jitter, and Retry-After header support
11. **Semantic re-ranking**: optional SPECTER2 embeddings for cosine similarity
    scoring in the screen stage
12. **Quality evaluation**: composite scoring — citation impact, venue reputation
    (CORE rankings), author h-index, recency bonus
13. **Two-tier conversion**: fast `convert-rough` (pymupdf4llm, all papers) then
    high-quality `convert-fine` (primary backend, selected papers)
14. **Incremental runs**: SQLite global paper index for cross-run dedup

## CLI entry point

The CLI is registered as `research-pipeline` in pyproject.toml:

```bash
# Full end-to-end pipeline
research-pipeline run "transformer architectures for time series"

# Individual stages
research-pipeline plan "topic"
research-pipeline search --run-id <ID>
research-pipeline screen --run-id <ID>
research-pipeline download --run-id <ID>
research-pipeline convert --run-id <ID>
research-pipeline extract --run-id <ID>
research-pipeline summarize --run-id <ID>

# Auxiliary commands
research-pipeline expand --run-id <ID> --direction both
research-pipeline quality --run-id <ID>
research-pipeline convert-rough --run-id <ID>
research-pipeline convert-fine --run-id <ID>
research-pipeline index --list

# Standalone PDF conversion (no workspace required)
research-pipeline convert-file paper.pdf -o paper.md

# Inspect run status
research-pipeline inspect --run-id <ID>
```

## MCP server

The MCP server wraps CLI logic into tools for AI agent integration:

```bash
# Run via module
python -m mcp_server

# Or via uv
uv run python -m mcp_server
```

## Adding a new pipeline stage

1. Create the domain model in `src/research_pipeline/models/`
2. Implement the logic module under `src/research_pipeline/<stage>/`
3. Add the CLI command in `src/research_pipeline/cli/cmd_<stage>.py`
4. Register the command in `src/research_pipeline/cli/app.py`
5. Add the stage to the orchestrator in `src/research_pipeline/pipeline/`
6. Add MCP tool in `mcp_server/tools.py` and schema in `mcp_server/schemas.py`
7. Write unit tests in `tests/unit/test_<stage>.py`
8. Run format, lint, type check, and tests before committing

## Adding a new converter backend

1. Create `src/research_pipeline/conversion/<name>_backend.py`
2. Subclass `ConverterBackend` from `conversion/base.py`
3. Decorate with `@register_backend("name")` from `conversion/registry.py`
4. Add config model (`*Account` + `*Config`) in `config/models.py`
5. Add account dispatch in `cli/cmd_convert.py` `_backend_kwargs_list()`
6. Add optional dependency in `pyproject.toml` extras
7. Write unit tests

## Adding a new source

1. Implement the `SearchSource` protocol from `src/research_pipeline/sources/base.py`
2. Register in the source factory
3. Add tests with VCR cassettes for HTTP mocking

## Common pitfalls

- Always use `uv run` to prefix Python commands (this is a uv-managed project)
- The `runs/` and `workspace/` directories are gitignored — don't track outputs
- arXiv rate limiting is global and thread-safe; don't bypass it
- Test fixtures directories may be empty in git — they're populated by tests
- The `config.toml` file is gitignored (may contain API keys); use
  `config.example.toml` as the template

## Commit and PR guidelines

- Commit message format: `<type>: <short description>` (e.g., `feat: add DOI
  lookup source`, `fix: handle empty search results`)
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`
- Always run `uv run pre-commit run --all-files` and
  `uv run pytest tests/unit/ -x -q` before committing
- Keep PRs focused on a single concern

## Related documentation

- [docs/architecture.md](docs/architecture.md) — system design, stage I/O
  contracts, cross-cutting concerns
- [docs/user-guide.md](docs/user-guide.md) — installation, configuration
  reference, CLI usage, MCP setup
- [config.example.toml](config.example.toml) — annotated configuration template
