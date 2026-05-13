# Developer Guide: research-pipeline

## 1. Document Purpose

This guide is for contributors and maintainers of the **research-pipeline**
project. It covers environment setup, development workflow, code conventions,
architecture overview, and step-by-step instructions for common extension tasks
(new stage, new converter backend, new search source, new MCP tool).

For end-user CLI/MCP reference see [user-guide.md](user-guide.md).
For system design and stage I/O contracts see [architecture.md](architecture.md).

---

## 2. Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Python | 3.12 | 3.13 also tested in CI |
| [uv](https://docs.astral.sh/uv/) | latest | replaces pip + venv + pip-tools |
| git | 2.x | standard version control |
| Docker *(optional)* | 20.x+ | only needed for containerised local runs |

> **Never** invoke bare `python`, `pytest`, or `pip` directly — always prefix
> with `uv run` (or activate the `.venv` explicitly).  The project uses
> `uv_build` as its build backend; plain `pip install -e .` is not supported.

---

## 3. Repository Setup

### 3.1 Clone and bootstrap

```bash
git clone https://github.com/grammy-jiang/research-pipeline.git
cd research-pipeline

# Install all dependencies (core + dev + docling + scholar extras)
uv sync --extra dev --extra docling --extra scholar
```

This creates `.venv/` and installs every tool needed for development (pytest,
ruff, mypy, pre-commit, etc.).

### 3.2 Optional extras

The project exposes optional dependency groups for converter backends and
additional sources:

| Extra | What it adds |
|-------|-------------|
| `docling` | Docling PDF→Markdown backend |
| `marker` | Marker PDF→Markdown backend |
| `pymupdf4llm` | Fast PyMuPDF4LLM backend (Tier 2 rough conversion) |
| `mineru` | MinerU (magic-pdf) backend |
| `mathpix` | Mathpix cloud OCR backend (no extra deps — API key only) |
| `datalab` | Datalab Marker cloud backend |
| `llamaparse` | LlamaParse cloud backend |
| `mistral-ocr` | Mistral OCR cloud backend (no extra deps) |
| `openai-vision` | OpenAI Vision backend (uses PyMuPDF for page images) |
| `scholar` | Google Scholar source via `scholarly` |
| `serpapi` | Google Scholar source via SerpAPI |
| `reranker` | SPECTER2 semantic re-ranking via `sentence-transformers` |

Install any combination:

```bash
uv sync --extra dev --extra pymupdf4llm --extra reranker
```

### 3.3 Verify installation

```bash
uv run research-pipeline --version
# Expected: research-pipeline 0.17.14

uv run python -c "import research_pipeline; print(research_pipeline.__version__)"
# Expected: 0.17.14
```

---

## 4. Development Workflow

### 4.1 Run tests

Unit tests are fast and require no network access:

```bash
# Run all unit tests (3700+) with verbose output
uv run pytest tests/unit/ -xvs

# Run a single test file
uv run pytest tests/unit/test_candidate.py -xvs

# Run a specific test function
uv run pytest tests/unit/test_adaptive_stopping.py::TestDetectKnee::test_clear_knee -xvs

# Run with coverage (CI threshold: 83%)
uv run pytest tests/unit/ -x -q --cov=src/research_pipeline --cov-report=term-missing
```

Integration tests use recorded VCR cassettes (no live network):

```bash
uv run pytest tests/integration_offline/ -xvs
```

Live tests hit real arXiv and are excluded from normal runs:

```bash
# Only when you have a network connection and want real data
uv run pytest tests/live/ -xvs -m live
```

### 4.2 Format and lint

```bash
# Auto-format (ruff format replaces Black)
uv run ruff format .

# Lint with auto-fix
uv run ruff check . --fix

# Lint check only (no writes — useful in CI)
uv run ruff check .
```

Ruff is configured in `pyproject.toml` with 88-character line length, targeting
Python 3.12.  Rule sets enabled: `A, B, C4, E, F, I, PT, RUF, SIM, UP, W`.

### 4.3 Type checking

```bash
uv run mypy src/
```

mypy runs in **strict mode** (`strict = true` in `pyproject.toml`).  CI
enforces zero errors.  The `mcp_server` sub-package is currently excluded from
mypy via `[[tool.mypy.overrides]]`.

### 4.4 Pre-commit hooks

Install once, then hooks run automatically on every `git commit`:

```bash
uv run pre-commit install
```

Run all hooks manually against the whole repository:

```bash
uv run pre-commit run --all-files
```

Hooks configured (from `.pre-commit-config.yaml`):

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with a newline |
| `check-yaml`, `check-toml`, `check-json` | Parse-level syntax checks |
| `check-added-large-files` | Block files > 1000 KB |
| `check-merge-conflict` | Detect leftover conflict markers |
| `check-case-conflict` | Catch case-only filename collisions |
| `check-symlinks` | Validate symlinks |
| `detect-private-key` | Prevent accidental secret commits |
| `debug-statements` | Reject `breakpoint()` / `pdb.set_trace()` |
| `check-ast` | Ensure Python files parse correctly |
| `name-tests-test` | Enforce `test_*.py` naming for test files |
| `check-docstring-first` | Module docstring must precede imports |
| `ruff-format` | Auto-format with ruff |
| `ruff` (with `--fix`) | Lint + auto-fix |
| `toml-sort-fix` | Keep TOML files sorted |
| `bandit` | Security static analysis (configured via `pyproject.toml`) |
| `validate-pyproject` | Validate `pyproject.toml` schema |
| `detect-secrets` | Secret scanning against `.secrets.baseline` |

### 4.5 CI pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs four jobs on every push and
pull request to `master`:

| Job | Command | Notes |
|-----|---------|-------|
| **Lint & Format** | `uv run pre-commit run --all-files` | Python 3.12 |
| **Test** | `uv run pytest tests/unit/ -x -q --cov=src/research_pipeline --cov-fail-under=83` | Python 3.12 and 3.13 |
| **Type Check** | `uv run mypy src/` | Strict mode, Python 3.12 |
| **Security & License Audit** | `uv run pip-audit` + `uv run pip-licenses` | Blocks GPL/AGPL dependencies |

---

## 5. Project Structure and Conventions

### 5.1 Repository layout

```
src/research_pipeline/       # Main library
  __init__.py                # Package root — re-exports __version__
  cli/                       # Typer CLI commands (one cmd_<stage>.py per stage)
  models/                    # Pydantic domain models (one file per concept)
  config/                    #   loader.py  models.py  defaults.py
  arxiv/                     # arXiv API client, XML parser, dedup
  sources/                   # Multi-source adapter (arXiv, Scholar, S2, OpenAlex, DBLP)
  screening/                 # BM25 scoring, SPECTER2 re-ranking, LLM judge
  download/                  # Rate-limited PDF downloader with retry
  conversion/                # PDF→Markdown backends + registry + fallback
  quality/                   # Citation, venue, author, composite scoring
  extraction/                # Markdown chunking, hybrid BM25+embedding retrieval
  summarization/             # Per-paper summaries, cross-paper synthesis
  pipeline/                  # Orchestrator, stage sequencing, TER loop
  storage/                   # Workspace dirs, manifests, SHA-256 hashing
  infra/                     # Cache, HTTP, logging, rate limiting, retry, entropy
  memory/                    # Multi-tier memory (working, episodic, KG, paging)
  llm/                       # LLM provider ABC + Ollama/OpenAI implementations
  analysis/                  # Claim decomposition and confidence scoring
  confidence/                # 4-layer confidence architecture
  evaluation/                # Dual metrics, horizon metric, blinding audit
  retrieval/                 # Hybrid BM25+embedding retrieval
  feedback/                  # User feedback store (ELO-style BM25 weight tuning)
  agents/                    # Multi-agent coordinator
  security/                  # MCP guard, adversarial tools, defense trilemma
  briefing/                  # Daily AI Intelligence pipeline
  mcp_server/                # FastMCP server (tools, schemas, resources, prompts)

tests/
  unit/                      # 3700+ fast, isolated unit tests (no network)
  integration_offline/       # VCR-cassette offline integration tests
  live/                      # Real-network tests (@pytest.mark.live)
  fixtures/                  # atom/, pdf/, markdown/, llm/, http_cassettes/

docs/                        # Project documentation (this file lives here)
pyproject.toml               # All project config (deps, tools, scripts)
config.example.toml          # Annotated config template (copy → config.toml)
.pre-commit-config.yaml      # Pre-commit hook definitions
uv.lock                      # Locked dependency graph
```

### 5.2 Code style rules

These are enforced by ruff and mypy.  Every violation blocks CI:

| Rule | Detail |
|------|--------|
| **Type hints** | Required on **all** function signatures — mypy strict mode |
| **Docstrings** | Google style, required on public functions and classes |
| **Formatter** | ruff format, 88-character line length |
| **Import sorting** | ruff (isort-compatible, configured via `[tool.ruff.lint]`) |
| **Path handling** | `pathlib.Path` everywhere — never `os.path` |
| **Domain models** | Pydantic `BaseModel` for all structured data |
| **Exception variable** | `as exc` — never `as e` |
| **f-strings** | Prefer over `.format()` or `%` formatting |
| **No mutable class defaults** | Pydantic models are exempt (`RUF012` ignored) |
| **Logging** | `logging` module only — `print()` is forbidden for operational output |

### 5.3 Logging and IO

```python
import logging

logger = logging.getLogger(__name__)

# Correct — module-level logger, standard levels
logger.info("Plan created: run_id=%s", run_id)
logger.debug("Query variants: %s", variants)
logger.error("Download failed: %s", exc)

# Wrong — never use print() for operational output
print("done")  # ❌
```

User-facing CLI output uses `typer.echo()`.  Everything else uses `logging`.
The log level is controlled by `--verbose` / `-v` on all commands.

### 5.4 Error handling

```python
try:
    result = do_something()
except SomeSpecificError as exc:   # ← always "exc", never "e"
    logger.error("Operation failed: %s", exc)
    raise
```

Use specific exception types.  Catch `Exception` only at the outermost boundary
(e.g., MCP tool wrappers) and always log the original error before converting it
to a user-friendly message.

---

## 6. Testing Guide

### 6.1 Test pyramid

| Layer | Location | Count | Network | When to run |
|-------|----------|-------|---------|------------|
| **Unit** | `tests/unit/` | 3700+ | ❌ none | Always — on every commit |
| **Integration (offline)** | `tests/integration_offline/` | ~12 | ❌ VCR | Before major merges |
| **Live** | `tests/live/` | small | ✅ arXiv | Explicitly (`-m live`) |

### 6.2 Test file naming conventions

Test files map directly to source modules:

```
src/research_pipeline/screening/adaptive_stopping.py
    →  tests/unit/test_adaptive_stopping.py

src/research_pipeline/models/candidate.py
    →  tests/unit/test_candidate.py

src/research_pipeline/conversion/registry.py
    →  tests/unit/test_conversion_registry.py
```

The `name-tests-test` pre-commit hook enforces the `test_*.py` prefix.

### 6.3 VCR cassettes

HTTP-dependent tests record real responses once and replay them from YAML
cassettes in `tests/fixtures/http_cassettes/`:

```python
import vcr

@vcr.use_cassette("tests/fixtures/http_cassettes/arxiv_search.yaml")
def test_arxiv_search_returns_candidates():
    ...
```

To record a new cassette: delete the cassette file and run the test with a live
network connection.  Commit the cassette alongside the test.

### 6.4 Test fixtures

Static test data lives in `tests/fixtures/`:

| Directory | Contents |
|-----------|----------|
| `atom/` | arXiv Atom XML feed responses |
| `pdf/` | Sample PDF files for conversion tests |
| `markdown/` | Pre-converted Markdown for extraction/summarization tests |
| `llm/` | Canned LLM responses for summarization tests |
| `http_cassettes/` | VCR cassette YAML files |

Fixture files are checked into git.  Do not add files larger than 1000 KB
(pre-commit blocks it).

### 6.5 Writing a new test

Follow TDD: write the test before writing the implementation.

```python
"""Tests for the new_feature module."""  # ← module docstring first

from __future__ import annotations

import pytest

from research_pipeline.new_module.new_feature import do_something


class TestDoSomething:
    def test_returns_expected_value(self) -> None:
        result = do_something("input")
        assert result == "expected"

    def test_raises_on_invalid_input(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            do_something("")
```

Key rules:
- Every test class/function name starts with `test_`.
- Use `pytest.raises` with a `match=` pattern rather than bare `with pytest.raises`.
- No network calls in unit tests — mock or use fixtures.
- Never modify existing tests without explicit approval.

---

## 7. Architecture for Contributors

### 7.1 Stage-based architecture

The pipeline has 7 sequential stages.  Each stage:
- reads from the previous stage's output directory
- writes its own artifacts to a named subdirectory under the run root
- records its status and artifact SHA-256 hashes in `run_manifest.json`
- is **idempotent** — re-running a completed stage produces the same output

```
plan → search → screen → download → convert → extract → summarize
 │        │        │         │          │          │          │
 ▼        ▼        ▼         ▼          ▼          ▼          ▼
query_   candidates/ shortlist/ pdfs/   markdown/ extracted/ summaries/
plan.json
```

Run root layout (inside `workspace/<run_id>/`):

```
plan/           query_plan.json
search/         candidates.jsonl
screen/         shortlist.jsonl, score_breakdown.jsonl
download/       download_manifest.json
convert/        convert_manifest.json
extract/        extraction_manifest.json, *.jsonl
summarize/      summaries/, synthesis_report.json, report.md
run_manifest.json
```

### 7.2 Pydantic models

All domain objects are Pydantic `BaseModel` subclasses in
`src/research_pipeline/models/`.  Use `model_dump_json()` and
`model_validate_json()` for serialisation:

```python
from pydantic import BaseModel, Field


class MyRecord(BaseModel):
    """A record for something new."""

    record_id: str = Field(description="Unique identifier.")
    value: float = Field(default=0.0, description="Numeric value.")
    tags: list[str] = Field(default_factory=list, description="Tag list.")
```

Rules:
- Every field must have a `description=` in its `Field(...)`.
- Optional fields use `X | None` with `default=None` (Python 3.10+ syntax).
- Do not use `Optional[X]` — use `X | None`.
- Use `default_factory=list` (or `dict`) for mutable defaults.

### 7.3 Configuration system

Configuration is loaded in order of precedence:
**environment variables > `config.toml` file > hard-coded defaults**

```python
from research_pipeline.config.loader import load_config

config = load_config()                  # auto-discover config.toml
config = load_config(Path("my.toml"))   # explicit path
```

Environment variables follow the pattern `RESEARCH_PIPELINE_<SECTION>_<KEY>`:

| Variable | Effect |
|----------|--------|
| `RESEARCH_PIPELINE_CONFIG` | Override config file path |
| `RESEARCH_PIPELINE_CACHE_DIR` | Override cache directory |
| `RESEARCH_PIPELINE_WORKSPACE` | Override workspace directory |
| `RESEARCH_PIPELINE_DISABLE_LLM` | Set `llm.enabled = false` |
| `RESEARCH_PIPELINE_LLM_PROFILE` | Override `llm.profile` |

New config options go in `src/research_pipeline/config/models.py` (Pydantic
schema) and `config.example.toml` (annotated template).  See
[Section 8.6](#86-add-a-new-config-option) for the full procedure.

### 7.4 CLI command pattern

Every CLI command follows the same three-layer pattern:

1. **`app.py`** — thin Typer command that calls `_common_options()` then
   delegates to the `run_*` function in the corresponding `cmd_*.py`.
2. **`cmd_<stage>.py`** — contains `run_<stage>(...)` which owns all business
   logic: load config, initialise workspace, call domain code, log results.
3. **Domain package** — pure logic, no Typer/CLI knowledge.

```python
# src/research_pipeline/cli/app.py  (registration)
@app.command()
def plan(
    topic: str = typer.Argument(..., help="Research topic."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Normalize a topic into a structured query plan."""
    from research_pipeline.cli.cmd_plan import run_plan

    opts = _common_options(verbose, config, workspace, run_id)
    run_plan(topic, **opts)
```

```python
# src/research_pipeline/cli/cmd_plan.py  (implementation)
def run_plan(
    topic: str,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the plan stage.

    Args:
        topic: Raw topic string.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Optional run ID.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id, run_root = init_run(ws, run_id)
    # ... domain logic ...
    logger.info("Plan saved to %s", plan_path)
```

### 7.5 MCP tool pattern

Every MCP tool is a three-layer stack:

1. **`server.py`** — `@mcp.tool(...)` decorated wrapper that accepts flat
   primitive arguments, constructs the typed input schema, and delegates to
   `tools.py`.
2. **`tools.py`** — pure adapter function that validates inputs, calls domain
   logic, and returns a `ToolResult`.
3. **`schemas.py`** — `CommonParams`-derived Pydantic schema for the tool's
   input.

```python
# server.py
@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_plan_topic(
    topic: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> dict:
    """Create a structured query plan from a natural language research topic."""
    result = plan_topic(
        PlanTopicInput(topic=topic, workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result.model_dump()
```

```python
# schemas.py
class PlanTopicInput(CommonParams):
    """Input for the plan_topic tool."""

    topic: str = Field(description="Natural language research topic.")
```

```python
# tools.py
def plan_topic(params: PlanTopicInput, ctx: Context | None = None) -> ToolResult:
    """Create a query plan from a natural language topic."""
    try:
        # ... domain logic ...
        return ToolResult(success=True, message="...", artifacts={...})
    except Exception as exc:
        logger.error("plan_topic failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")
```

---

## 8. How-to Guides

### 8.1 Add a new pipeline stage

Follow these steps in order.  Run tests after each step.

**Step 1 — Domain model**

Create `src/research_pipeline/models/<stage>.py`:

```python
"""Domain model for the <stage> stage."""

from pydantic import BaseModel, Field


class MyStageRecord(BaseModel):
    """Output record from the <stage> stage."""

    record_id: str = Field(description="Unique identifier.")
    result: str = Field(description="Stage output.")
```

**Step 2 — Logic package**

Create `src/research_pipeline/<stage>/__init__.py` and implement the stage logic.
Import models; never import CLI or MCP modules.

**Step 3 — CLI command**

Create `src/research_pipeline/cli/cmd_<stage>.py`:

```python
"""CLI handler for the '<stage>' command."""

import logging
from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_<stage>(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the <stage> stage.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run identifier.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id, run_root = init_run(ws, run_id)
    stage_dir = get_stage_dir(run_root, "<stage>")
    # ... stage logic ...
    logger.info("<stage> complete: run_id=%s", run_id)
```

**Step 4 — Register in app.py**

Add to `src/research_pipeline/cli/app.py` (following the existing pattern):

```python
@app.command()
def my_stage(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """One-line description of this stage."""
    from research_pipeline.cli.cmd_my_stage import run_my_stage

    opts = _common_options(verbose, config, workspace, run_id)
    run_my_stage(**opts)
```

**Step 5 — Add to orchestrator**

Register the stage in `src/research_pipeline/pipeline/` so that
`research-pipeline run` executes it in order.

**Step 6 — MCP tool**

Add input schema to `src/research_pipeline/mcp_server/schemas.py`:

```python
class MyStageInput(CommonParams):
    """Input for the my_stage tool."""
    pass
```

Add tool function to `src/research_pipeline/mcp_server/tools.py`:

```python
def my_stage_tool(params: MyStageInput, ctx: Context | None = None) -> ToolResult:
    """Run the <stage> stage."""
    try:
        from research_pipeline.cli.cmd_my_stage import run_my_stage
        run_my_stage(workspace=Path(params.workspace), run_id=params.run_id or None)
        return ToolResult(success=True, message="Stage complete.")
    except Exception as exc:
        logger.error("my_stage_tool failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")
```

Register in `src/research_pipeline/mcp_server/server.py` using `@mcp.tool(...)`.

**Step 7 — Tests**

Create `tests/unit/test_<stage>.py` and write tests before implementing the logic.

```bash
uv run pytest tests/unit/test_my_stage.py -xvs
```

### 8.2 Add a new converter backend

**Step 1 — Backend class**

Create `src/research_pipeline/conversion/<name>_backend.py`:

```python
"""<Name> PDF→Markdown converter backend."""

import logging
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("mybackend")
class MyBackendConverter(ConverterBackend):
    """Convert PDFs to Markdown using MyBackend."""

    def __init__(self, api_key: str) -> None:
        """Initialise backend.

        Args:
            api_key: API key for the service.
        """
        self._api_key = api_key

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        """Convert a single PDF.

        Args:
            pdf_path: Source PDF.
            output_dir: Output directory for Markdown.
            force: Re-convert even if output exists.

        Returns:
            Manifest entry with result metadata.
        """
        # ... conversion logic ...

    def fingerprint(self) -> str:
        """Return deterministic backend fingerprint."""
        return "mybackend/1.0/default"
```

**Step 2 — Config model**

Add to `src/research_pipeline/config/models.py`:

```python
class MyBackendConfig(BaseModel):
    """Configuration for the MyBackend converter."""

    api_key: str = Field(default="", description="API key.")
```

Then add `mybackend: MyBackendConfig = Field(default_factory=MyBackendConfig)`
to the `ConversionConfig` model.

**Step 3 — Account dispatch**

Add a branch in `cli/cmd_convert.py` `_backend_kwargs_list()` that maps config
to constructor kwargs for your backend.

**Step 4 — Optional dependency**

Add to `[project.optional-dependencies]` in `pyproject.toml`:

```toml
mybackend = ["mybackend-sdk>=1.0"]
```

**Step 5 — Tests**

```python
# tests/unit/test_mybackend_backend.py
from research_pipeline.conversion.mybackend_backend import MyBackendConverter

def test_fingerprint_is_stable() -> None:
    converter = MyBackendConverter(api_key="test")
    assert converter.fingerprint() == "mybackend/1.0/default"
```

### 8.3 Add a new search source

**Step 1 — Implement the protocol**

Create `src/research_pipeline/sources/<name>_source.py` and implement the
`SearchSource` protocol from `src/research_pipeline/sources/base.py`:

```python
"""<Name> search source adapter."""

import logging

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.base import SearchSource

logger = logging.getLogger(__name__)


class MySearchSource:
    """Search adapter for MySource."""

    @property
    def name(self) -> str:
        """Return source name."""
        return "mysource"

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Execute search and return candidate records.

        Args:
            topic: Raw topic string.
            must_terms: High-priority terms.
            nice_terms: Boosting terms.
            max_results: Result cap.
            date_from: Start of date window (ISO 8601).
            date_to: End of date window (ISO 8601).

        Returns:
            Deduplicated candidate records.
        """
        # ... API call + parsing ...
        return []
```

**Step 2 — Register in source factory**

Add your source to the factory in `src/research_pipeline/sources/` so that
`--source mysource` and `--source all` include it.

**Step 3 — VCR tests**

Record an HTTP cassette and write tests:

```python
import vcr

@vcr.use_cassette("tests/fixtures/http_cassettes/mysource_search.yaml")
def test_mysource_returns_candidates() -> None:
    source = MySearchSource()
    results = source.search("transformers", ["transformer"], [], max_results=5)
    assert len(results) > 0
    assert all(r.source == "mysource" for r in results)
```

### 8.4 Add a new MCP tool

**Step 1 — Schema**

In `src/research_pipeline/mcp_server/schemas.py`:

```python
class MyNewToolInput(CommonParams):
    """Input for the my_new_tool tool."""

    some_param: str = Field(description="Description of the parameter.")
    optional_param: int = Field(default=10, description="Optional integer.")
```

**Step 2 — Tool function**

In `src/research_pipeline/mcp_server/tools.py`:

```python
def my_new_tool(params: MyNewToolInput, ctx: Context | None = None) -> ToolResult:
    """Execute my new tool."""
    try:
        # ... logic, calling domain code ...
        _log_info(ctx, f"my_new_tool complete: {params.some_param}")
        return ToolResult(success=True, message="Done.", artifacts={})
    except Exception as exc:
        logger.error("my_new_tool failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")
```

**Step 3 — Registration**

In `src/research_pipeline/mcp_server/server.py`, add the import and the
`@mcp.tool(...)` wrapper:

```python
from research_pipeline.mcp_server.tools import my_new_tool
from research_pipeline.mcp_server.schemas import MyNewToolInput

@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_my_new_tool(
    some_param: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    optional_param: int = 10,
) -> dict:
    """Short description shown in MCP tool list."""
    result = my_new_tool(
        MyNewToolInput(
            some_param=some_param,
            workspace=workspace,
            run_id=run_id,
            optional_param=optional_param,
        ),
        ctx=ctx,
    )
    return result.model_dump()
```

### 8.5 Add a new CLI command

For standalone utility commands that don't map to a pipeline stage, add them
directly in `app.py` rather than creating a `cmd_*.py` file — unless the
implementation is more than ~30 lines, in which case extract to `cmd_<name>.py`.

All commands **must** accept `--verbose`, `--config`, and `--workspace` via
`_common_options()` unless there is a strong reason not to.

### 8.6 Add a new config option

1. Add the field to the relevant Pydantic model in
   `src/research_pipeline/config/models.py`:

   ```python
   class SearchConfig(BaseModel):
       # ... existing fields ...
       my_new_option: int = Field(
           default=42,
           description="Description shown in docs.",
       )
   ```

2. Add an annotated entry to `config.example.toml`:

   ```toml
   [search]
   # my_new_option: description (default: 42)
   my_new_option = 42
   ```

3. If the option should be overridable from the environment, add a mapping in
   `src/research_pipeline/config/loader.py` `_apply_env_overrides()`.

4. Write a unit test in `tests/unit/test_config_models.py` or the appropriate
   test file.

---

## 9. Hard Constraints (HC1–HC6)

These constraints apply to every contributor and every automated agent.  They
are defined in `AGENTS.md` and cannot be relaxed by any runtime overlay.

| ID | Rule | Enforcement |
|----|------|-------------|
| **HC1** | No plaintext secrets in repository files, prompts, logs, or commits | `detect-private-key` + `detect-secrets` pre-commit hooks; `.secrets.baseline` required |
| **HC2** | No writes outside the allowlist: `src/`, `tests/`, `docs/`, `pyproject.toml`, `.pre-commit-config.yaml`, `Makefile`, `AGENTS.md`, `CLAUDE.md`, `.github/` | Reviewed in code review; automated agents must deny out-of-scope writes |
| **HC3** | Destructive commands (`rm -rf`, `git push --force`, `git reset --hard`, `DROP TABLE`) require explicit human approval before execution | Documented process; CI never runs destructive ops autonomously |
| **HC4** | Database schema changes (migrations, drops) must be authored but never executed autonomously | Reviewed before merge; no auto-apply in CI |
| **HC5** | Network egress limited to: `arxiv.org`, `export.arxiv.org`, `api.semanticscholar.org`, `api.openalex.org`, `dblp.org`, `serpapi.com`, `pypi.org`, `files.pythonhosted.org`, `github.com` | All other destinations require explicit human approval |
| **HC6** | Red-class data (secrets, PII, credentials, API keys, session tokens) must never enter prompts, tool arguments, trace logs, plan files, or stored artefacts | API keys belong in `config.toml` (gitignored) or environment variables only |

`config.toml` is listed in `.gitignore` and **must never be committed**.  Use
`config.example.toml` as the template.  The `.secrets.baseline` file must be
kept up-to-date when new secrets patterns are introduced.

---

## 10. Dependency Management

The project uses **uv** as the sole package manager.  `pip`, `pip-tools`, and
`poetry` are not used.

### Key files

| File | Purpose |
|------|---------|
| `pyproject.toml` | All dependency declarations, tool config, scripts |
| `uv.lock` | Deterministic locked dependency graph (commit this) |
| `.python-version` | Pins Python to 3.12 for local development |

### Adding a dependency

```bash
# Add a runtime dependency
uv add some-package

# Add a dev-only dependency
uv add --dev some-package

# Add to an optional extra
uv add --optional docling some-new-package
```

After adding, commit both `pyproject.toml` and `uv.lock`.

### Updating dependencies

```bash
# Update all packages (within constraints)
uv sync --upgrade

# Update a single package
uv add some-package>=new-version
```

### Security audit

```bash
uv run pip-audit --skip-editable
uv run pip-licenses --fail-on="GPL-3.0-only;AGPL-3.0-only"
```

Both commands run in the `security` CI job.

---

## 11. Release Process

Releases are triggered by creating a GitHub Release (tag format: `vX.Y.Z`).
The `publish` workflow (`publish.yml`) runs automatically:

1. **Build** — `uv build` produces wheel and sdist in `dist/`
2. **Publish to PyPI** — via `pypa/gh-action-pypi-publish` using trusted
   publishing (OIDC, no API token required)
3. **Attach to GitHub Release** — wheel and sdist uploaded via `gh release upload`

Pre-release checklist:

```bash
# 1. Bump version in pyproject.toml
#    (version is read at runtime via importlib.metadata)

# 2. Update CHANGELOG.md

# 3. Run full quality gate
uv run pre-commit run --all-files
uv run pytest tests/unit/ -x -q --cov=src/research_pipeline --cov-fail-under=83
uv run mypy src/

# 4. Tag and push (triggers CI; CI must be green before creating the Release)
git tag vX.Y.Z
git push origin vX.Y.Z
```

Creating the GitHub Release (not just the tag) is what triggers the publish
workflow.

---

## 12. Getting Help and Contributing

- **Bug reports / feature requests**: [GitHub Issues](https://github.com/grammy-jiang/research-pipeline/issues)
- **Documentation**: [grammy-jiang.github.io/research-pipeline](https://grammy-jiang.github.io/research-pipeline/)
- **Source**: [github.com/grammy-jiang/research-pipeline](https://github.com/grammy-jiang/research-pipeline)

### Contribution flow

1. Fork the repository and create a feature branch.
2. Write tests first (TDD).
3. Implement the feature, following all code style rules.
4. Run the full quality gate (pre-commit + tests + mypy).
5. Open a pull request against `master` with a clear description.
6. Ensure all CI jobs pass.

### Commit message format

```
<type>: <short description>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`.

Examples:
```
feat: add MinerU converter backend
fix: handle empty abstract in CandidateRecord
test: add VCR cassette for OpenAlex pagination
docs: document new config options in developer-guide
```
