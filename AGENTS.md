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
│   ├── copilot-instructions.md   # GitHub Copilot instructions
│   ├── instructions/             # Path-specific coding instructions
│   └── workflows/                # CI workflows
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
│   ├── screening/                # Heuristic BM25 scoring, SPECTER2 embeddings, LLM judge, depth gate
│   ├── download/                 # Rate-limited PDF downloader with retry
│   ├── conversion/               # PDF→Markdown backends (3 local + 5 cloud + fallback)
│   ├── quality/                  # Quality evaluation (citations, venue, author, composite, safety gate, RACE, graduated criteria)
│   ├── extraction/               # Markdown chunking & hybrid BM25+embedding retrieval
│   ├── summarization/            # Per-paper + cross-paper synthesis
│   ├── pipeline/                 # Orchestrator, stage sequencing, enhanced checkpoints, plan revision, adaptive topology
│   ├── storage/                  # Workspace, manifests, artifacts, global index
│   ├── infra/                    # Cache, HTTP, logging, hashing, clock, rate limiting, retry, sanitization, audit, entropy monitor
│   ├── memory/                   # Multi-tier memory system (working, episodic, semantic, CMA audit, associative, paging)
│   ├── security/                 # MCP guard, adversarial robustness (ToolTweak), defense trilemma
│   ├── mcp_server/               # Packaged FastMCP server
│   │   ├── server.py             # Server entry point, registrations
│   │   ├── tools.py              # MCP tool implementations
│   │   ├── schemas.py            # Input/output Pydantic schemas
│   │   └── workflow/             # Harness-engineered research workflow
│   ├── skill_data/               # Bundled skill files for pip package
│   ├── agent_data/               # Bundled sub-agent definitions for pip package
│   ├── llm/                      # LLM provider interface + providers
│   │   ├── base.py               # Abstract LLMProvider ABC
│   │   ├── providers.py          # Ollama + OpenAI-compatible implementations
│   │   ├── schemas.py            # Pydantic I/O schemas for LLM calls
│   │   └── envelopes.py          # Input/output envelope wrappers
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
# Format & lint (ruff handles both)
uv run ruff format .
uv run ruff check . --fix

# Type checking (strict mode, 0 errors enforced in CI)
uv run mypy src/
```

Pre-commit hooks enforce this automatically:

```bash
uv run pre-commit run --all-files
```

## Code style

- **Formatter**: ruff format (88 char line length)
- **Import sorting**: ruff (isort-compatible)
- **Linter**: ruff (select: A, B, C4, E, F, I, PT, RUF, SIM, UP, W)
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
   Plus 8 auxiliary commands: expand, quality, convert-rough, convert-fine,
   index, analyze, validate, compare
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
    (CORE rankings), author h-index, recency bonus, reproducibility
13. **Two-tier conversion**: fast `convert-rough` (pymupdf4llm, all papers) then
    high-quality `convert-fine` (primary backend, selected papers)
14. **Incremental runs**: SQLite global paper index for cross-run dedup
15. **Verification gates**: structural verification for every pipeline stage
    output (plan, search, screen, download, convert, extract, summarize)
16. **Diversity-aware screening**: MMR-style selection balances relevance with
    category, source, and year diversity in shortlist construction
17. **Report validation**: 14-section template compliance checking with
    confidence-level, citation, gap classification, and formatting checks
18. **Cross-run comparison**: structured diff of papers, gaps, confidence
    changes, readiness, and quality scores between pipeline runs
19. **Multi-tier memory**: working memory (bounded deque), episodic memory
    (SQLite), semantic KG, CMA six-property audit, A-MEM associative linking,
    MemGPT-style tiered paging with fault counters
20. **Adaptive topology**: DifficultyRouter→PipelineProfile mapping for
    automatic pipeline configuration by query complexity
21. **Plan revision**: TER-based plan-revision scoring with plateau detection
    for iterative research loops
22. **Entropy monitoring**: Shannon entropy rolling-window monitor for
    token-level drift and in-context locking detection
23. **Adversarial robustness**: ToolTweak 10-perturbation catalog for tool
    schema fuzz testing and defense-trilemma K^n budget tracking
24. **Long-horizon failure modes**: UltraHorizon 8-mode taxonomy for detecting
    failure patterns in multi-step research workflows

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

# Quality & analysis commands
research-pipeline analyze --run-id <ID>        # Prepare per-paper analysis tasks
research-pipeline analyze --run-id <ID> --collect  # Validate collected analysis results
research-pipeline validate --report report.md   # Validate report completeness
research-pipeline compare --run-a <ID1> --run-b <ID2>  # Cross-run comparison
research-pipeline horizon --score 0.8 --achieved 40 --target 50  # Unified Horizon Metric (A3-5)
research-pipeline rrp --report report.md --shortlist shortlist.json  # R/R/P diagnostic (Theme 16)

# Auxiliary commands
research-pipeline expand --run-id <ID> --direction both
research-pipeline expand --run-id <ID> --paper-ids "ID1,ID2" --snowball --bfs-query "term1,term2"
research-pipeline quality --run-id <ID>
research-pipeline convert-rough --run-id <ID>
research-pipeline convert-fine --run-id <ID>
research-pipeline index --list
research-pipeline feedback --run-id <ID> --accept <PAPER_ID> --reject <PAPER_ID>
research-pipeline feedback --run-id <ID> --show --adjust

# Three-channel eval logging inspection
research-pipeline eval-log --run-id <ID>
research-pipeline eval-log --run-id <ID> --channel traces --stage screen
research-pipeline eval-log --run-id <ID> --channel summary

# Evidence-only aggregation
research-pipeline aggregate --run-id <ID>
research-pipeline aggregate --run-id <ID> --min-pointers 1 --format json

# HTML report export
research-pipeline export-html --run-id <ID>
research-pipeline export-html --markdown report.md -o report.html

# Standalone PDF conversion (no workspace required)
research-pipeline convert-file paper.pdf -o paper.md

# Inspect run status
research-pipeline inspect --run-id <ID>
```

## MCP server

The MCP server provides full Model Context Protocol support:

```bash
research-pipeline mcp serve
uv run research-pipeline mcp serve  # development checkout
```

Features: 51 tools (with annotations & progress), 15 resources (URI templates),
6 prompts, auto-completions, harness-engineered research workflow.

New quality tools:
- `analyze_claims` — decompose summaries into atomic claims
- `score_claims` — score confidence for decomposed claims
- `kg_stats` — knowledge graph entity/triple statistics
- `kg_query` — query entity + relations in KG
- `kg_ingest` — ingest pipeline results into KG
- `memory_stats` — memory tier statistics
- `memory_episodes` — list recent episodic memories
- `memory_search` — search episodic memory by topic
- `evaluate` — validate pipeline outputs against schemas
- `analyze_papers` — prepare per-paper analysis tasks or validate results
- `validate_report` — check report completeness (14 sections, citations, gaps)
- `compare_runs` — structured diff between two pipeline runs
- `verify_stage` — structural verification gates for any pipeline stage
- `export_html` — render synthesis report as self-contained HTML
- `model_routing_info` — inspect phase-aware model routing configuration
- `gate_info` — inspect HITL gate configuration
- `tool_coherence` — evaluate multi-session coherence across runs
- `tool_consolidation` — consolidate cross-run memory (episodes→rules, pruning, drift)

The `research_workflow` tool provides server-driven orchestration with 6 harness
layers: telemetry, context engineering, governance, structural verification,
doom-loop monitoring, and crash recovery. See `docs/user-guide.md` for the full
reference.

## AI skill & sub-agents

Install the bundled Claude Code / GitHub Copilot skill and sub-agent definitions:

```bash
research-pipeline setup              # Copy skills, agents, and MCP config
research-pipeline setup --symlink    # Symlink for development
research-pipeline setup --force      # Force overwrite existing
research-pipeline setup --skip-agents  # Skill only
research-pipeline setup --skip-skill   # Agents only
research-pipeline setup --skip-mcp     # Do not write MCP config snippet
```

Setup auto-detects which agents are installed and only writes to paths for
agents found on PATH:

| Agent | Skill path | Agents path | MCP registration |
|-------|-----------|-------------|-----------------|
| Claude Code | `~/.claude/skills/research-pipeline/` | `~/.claude/agents/*.md` | `claude mcp add` (user scope) |
| Codex CLI | `~/.agents/skills/research-pipeline/` | — (skills only) | `codex mcp add` |
| GitHub Copilot CLI | `~/.copilot/skills/research-pipeline/` | `~/.copilot/agents/*.agent.md` | `~/.copilot/mcp-config.json` |

MCP config snippet → `~/.config/research-pipeline/mcp.json`

`setup` auto-discovers every bundled skill (any `skill_data/*/SKILL.md`)
and fans out each one into its own directory under the agent's skills path.
Four skills ship today:

| Skill | Purpose |
|-------|---------|
| `research-pipeline` | Academic literature search → screen → convert → synthesize report |
| `daily-ai-intelligence` | Private daily AI tooling intelligence brief |
| `blueprint` | Convert a research report into an implementation-neutral product blueprint (`<topic-slug>-product-blueprint.md`) |
| `architecture` | Convert a product blueprint into a concrete technical architecture and tech-stack design (`<topic-slug>-architecture-design.md`) |

The `blueprint` skill is a pure prompt-driven transformation (no CLI/MCP
backend): it classifies input quality, maps `ACADEMIC`/`ENGINEERING` gaps
to validation requirements vs product requirements, resolves each idea as
ADOPT/ADAPT/MERGE/DEFER/REJECT, and emits an 18-section blueprint that
stays tech-stack-neutral and hands off to the `architecture` skill.

The `architecture` skill is also pure prompt-driven (no CLI/MCP backend) and
continues the chain `research-pipeline → blueprint → architecture →
implementation-plan`. It discovers/parses a product blueprint, builds a
blueprint-to-architecture traceability map, runs a tech-stack/AI-boundary
co-design loop, and emits a 25-section architecture document with C4 views, a
Traditional-vs-AI responsibility matrix, interface and data contracts,
security/trust boundaries, observability/audit, ADRs, and
implementation-planning handoff notes. It selects a tech stack with rationale
but never writes code or implementation tasks, keeps durable state under
deterministic control (AI output is validated before any state change), adopts
MCP only when justified, and supports regenerate/patch/compare/adr-only/resume
update modes with an `## Update History`.

The skills are structured per Anthropic's Skill-Building Guide (explicit
trigger phrases, standard-only `name`/`description`/`license` frontmatter,
progressive disclosure into `references/`). `research-pipeline` core
behaviors: **resume on
top of any prior same-topic report** (snapshot-rename, seed with prior
paper IDs, regenerate from scratch — never append); **iterate up to 4
gap-closure rounds** (stop on empty gap list / no-new-papers /
out-of-scope); and **enforce report formatting** (`## Contents`,
`## Round History`, Mermaid charts, LaTeX formulas, evidence-cited
findings validated by `research-pipeline validate`).

## Adding a new pipeline stage

1. Create the domain model in `src/research_pipeline/models/`
2. Implement the logic module under `src/research_pipeline/<stage>/`
3. Add the CLI command in `src/research_pipeline/cli/cmd_<stage>.py`
4. Register the command in `src/research_pipeline/cli/app.py`
5. Add the stage to the orchestrator in `src/research_pipeline/pipeline/`
6. Add MCP tool in `src/research_pipeline/mcp_server/tools.py` and schema in `src/research_pipeline/mcp_server/schemas.py`
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

## Hard Constraints

These constraints apply to every agent session in this repository. They must
never be relaxed by runtime overlays (`CLAUDE.md`, `.github/copilot-instructions.md`,
`.codex/INSTRUCTIONS.md`).

| ID | Rule |
|----|------|
| **HC1** | No plaintext secrets in repository files, prompts, logs, or commits. `detect-secrets` pre-commit hook and `.secrets.baseline` are required. |
| **HC2** | No agent-authored writes outside the path allowlist (`src/`, `tests/`, `docs/`, `pyproject.toml`, `.pre-commit-config.yaml`, `Makefile`, `AGENTS.md`, `CLAUDE.md`, `.github/`). Out-of-scope writes must be denied and reverted. |
| **HC3** | Destructive commands (`rm -rf`, `git push --force`, `git reset --hard`, `DROP TABLE`) require explicit human approval before execution. |
| **HC4** | Database schema changes (migrations, drops) must be authored but never executed autonomously. |
| **HC5** | Network egress from agent-executed code is limited to: `arxiv.org`, `export.arxiv.org`, `api.semanticscholar.org`, `api.openalex.org`, `dblp.org`, `serpapi.com`, `pypi.org`, `files.pythonhosted.org`, `github.com`. All other destinations require explicit human approval. |
| **HC6** | Red-class data (secrets, PII, credentials, API keys, session tokens) must never enter prompts, tool arguments, trace logs, plan files, or stored artefacts. API keys belong in `config.toml` (gitignored) or environment variables only. |

## Commit and PR guidelines

- Commit message format: `<type>: <short description>` (e.g., `feat: add DOI
  lookup source`, `fix: handle empty search results`)
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`
- Always run `uv run pre-commit run --all-files` and
  `uv run pytest tests/unit/ -x -q` before committing
- Keep PRs focused on a single concern

## Related documentation

- [docs/system-design.md](docs/system-design.md) — system design, stage I/O
  contracts, cross-cutting concerns
- [docs/implementation-plan.md](docs/implementation-plan.md) — current state,
  open work, design-document follow-through, out-of-scope items
- [docs/user-guide.md](docs/user-guide.md) — installation, configuration
  reference, CLI usage, MCP setup
- [docs/developer-guide.md](docs/developer-guide.md) — contributor onboarding,
  adding stages/backends/sources
- [docs/api-reference.md](docs/api-reference.md) — all CLI commands and MCP
  tools/resources/prompts
- [docs/data-model.md](docs/data-model.md) — Pydantic domain models and SQLite schemas
- [docs/security-model.md](docs/security-model.md) — HC1–HC6 constraints, MCP
  guard, taint tracking
- [docs/testing-strategy.md](docs/testing-strategy.md) — test pyramid, VCR
  cassettes, CI gates
- [docs/operations-runbook.md](docs/operations-runbook.md) — deploy, monitor,
  troubleshoot, maintain
- [docs/adr/](docs/adr/) — Architecture Decision Records
- [docs/changelog.md](docs/changelog.md) — versioned history
- [docs/audit-deep-research-compliance-2026-05-17.md](docs/audit-deep-research-compliance-2026-05-17.md) — latest compliance audit
- [config.example.toml](config.example.toml) — annotated configuration template
