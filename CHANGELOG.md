# Changelog

All notable changes to research-pipeline.

## [v0.14.3] — 2026-04-18

### Added

- 64 new unit tests covering 16 previously untested CLI handlers and scholar source
- CI coverage threshold raised from 80% to 85%

### Fixed

- Remove deprecated PEP 639 `License :: OSI Approved :: MIT License` classifier
  (`license = "MIT"` SPDX identifier is sufficient)

### Changed

- Coverage: 80% → 86% (3430 tests, 15816 statements, only `__main__.py` at 0%)

## [v0.14.2] — 2026-04-18

### Added

- 98 new unit tests across 7 files covering 20+ previously untested modules
- CI coverage threshold raised from 75% to 80%
- CI coverage report XML artifact upload (Python 3.12)
- README badges: CI status, mypy, ruff
- GitHub repo description and 10 topics
- PyPI classifiers: Scientific/Engineering, MIT License

### Changed

- Development Status classifier upgraded: Alpha → Beta
- Coverage: 77% → 80% (3362 tests, 15816 statements)

## [v0.14.1] — 2026-04-18

### Fixed

- Fix all 241 mypy errors across 59 source files (now 0 errors in 211 files)
- Fix real bugs found via type checking: non-existent `config.runs_dir`,
  `resolve_workspace`, `LLMProvider.complete()`, wrong `get_stage_dir` args
- Fix variable type conflicts from shadowing in 5 modules
- Enforce mypy in CI (removed `|| true` fallback)

### Added

- PEP 561 `py.typed` marker for downstream type checking
- CLI smoke tests (22 subcommands verified via `--help`)
- Coverage threshold `--cov-fail-under=70` in CI
- Expanded ruff rules: A, C4, PT, RUF groups

## [v0.14.0] — 2026-04-18

### Changed

- Replace Black + isort with Ruff format/check (single toolchain)
- Modernize pre-commit hooks: add bandit, validate-pyproject, remove legacy hooks

### Added

- pip-audit and pip-licenses for security & license auditing
- vulture for dead code detection
- hypothesis for property-based testing (36 tests)
- Security & License Audit CI job
- Python 3.13 added to CI test matrix

## [v0.13.52] — 2026-04-18

### Added

- MCP tool wrappers for 6 new commands (cite-context, cluster, export-bibtex,
  eval-log, feedback, export-html) with integration tests (C2+C3)

## [v0.13.51] — 2026-04-18

### Added

- CHANGELOG.md auto-generated from git history (C5)

## [v0.13.50] — 2026-04-18

### Documentation

- update user-guide with 6 new CLI commands (v0.13.44-49)

## [v0.13.49] — 2026-04-18

### Added

- watch mode for topics (B8) — periodic new paper monitoring

## [v0.13.48] — 2026-04-18

### Added

- citation context extraction (B7) — in-text citation sentence extraction

## [v0.13.47] — 2026-04-18

### Added

- abstract enrichment pipeline (B6) — DOI + title-based S2 lookup

## [v0.13.46] — 2026-04-18

### Added

- paper similarity clustering (B5) — TF-IDF + agglomerative clustering

## [v0.13.45] — 2026-04-18

### Added

- report template system with 4 built-in formats (B4)

## [v0.13.44] — 2026-04-18

### Added

- BibTeX export from candidate records (B2)

## [v0.13.43] — 2026-04-18

### Fixed

- sync black/ruff versions in pre-commit, use pre-commit in CI, track skill config.toml

## [v0.13.42] — 2026-04-18

### Fixed

- set black target-version py312, add jinja2 to dev deps

## [v0.13.41] — 2026-04-18

### CI

- add GitHub Actions CI workflow (lint, test, typecheck)

## [v0.13.40] — 2026-04-18

### Added

- full multi-source parallel search + incremental runs via global index

## [v0.13.39] — 2026-04-17

### Added

- wire 5 standalone modules into pipeline flow (Tier A integration)

## [v0.13.38] — 2026-04-17

### Added

- add 7-Dimension Coherence Evaluation framework

## [v0.13.37] — 2026-04-17

### Added

- add Scientific KG Benchmark framework

## [v0.13.36] — 2026-04-17

### Added

- add RL query reformulation with Thompson sampling bandit

## [v0.13.35] — 2026-04-17

### Added

- add adaptive difficulty routing (v0.13.35)

## [v0.13.34] — 2026-04-17

### Added

- add multi-model consensus engine (v0.13.34)

## [v0.13.33] — 2026-04-17

### Added

- add query-typed retrieval stopping profiles (v0.13.33)

## [v0.13.32] — 2026-04-17

### Added

- forward citation traversal with budget-aware stopping

## [v0.13.31] — 2026-04-17

### Added

- pre-commitment protocol for conformity bias elimination

## [v0.13.30] — 2026-04-17

### Added

- retention regularization — drift detection and score penalty

## [v0.13.29] — 2026-04-17

### Added

- add full environment snapshot capture at stage boundaries (v0.13.29)

## [v0.13.28] — 2026-04-17

### Added

- add graduated rubric scoring with 4-level grades (v0.13.28)

## [v0.13.27] — 2026-04-17

### Added

- add hash-pinned tool definitions for MCP integrity (v0.13.27)

## [v0.13.26] — 2026-04-17

### Added

- add length normalization for LLM responses (v0.13.26)

## [v0.13.25] — 2026-04-17

### Added

- claim-level citation accuracy scoring (v0.13.25)

## [v0.13.24] — 2026-04-17

### Added

- non-destructive versioned memory with rollback (v0.13.24)

## [v0.13.23] — 2026-04-17

### Added

- failure taxonomy logging with JSONL persistence (v0.13.23)

## [v0.13.22] — 2026-04-17

### Added

- structured output enforcement for LLM responses (v0.13.22)

## [v0.13.21] — 2026-04-17

### Added

- segment-level memory entries with token-aware splitting

## [v0.13.20] — 2026-04-17

### Added

- Q2D query augmentation with domain synonym expansion

## [v0.13.19] — 2026-04-17

### Added

- citation budget stopping criteria for BFS expansion

## [v0.13.18] — 2026-04-17

### Added

- query noise removal with academic boilerplate filtering

## [v0.13.17] — 2026-04-17

### Added

- heuristic dissent preservation in template-mode synthesis

## [v0.13.16] — 2026-04-17

### Added

- true MMR diversity with Jaccard document similarity in screening

## [v0.13.15] — 2026-04-17

### Added

- add 4-layer confidence architecture (C4)

## [v0.13.14] — 2026-04-17

### Added

- add query-adaptive retrieval stopping criteria (C3)

## [v0.13.13] — 2026-04-17

### Added

- add KG quality evaluation framework (5-dimension, 3-layer)

## [v0.13.12] — 2026-04-17

### Added

- add Case-Based Reasoning (CBR) for strategy reuse

## [v0.13.11] — 2026-04-17

### Added

- add Pass@k + Pass[k] dual metrics evaluation (B6)

## [v0.13.10] — 2026-04-17

### Added

- add epistemic blinding audits (B5)

## [v0.13.9] — 2026-04-17

### Added

- memory consolidation engine (B4)

## [v0.13.8] — 2026-04-16

### Added

- multi-session coherence evaluation (B3)

## [v0.13.7] — 2026-04-16

### Added

- human-in-the-loop approval gates (B2)

## [v0.13.6] — 2026-04-16

### Added

- phase-aware model routing (B1)

## [v0.13.5] — 2026-04-16

### Fixed

- correct config access and synthesis filename in aggregate/export-html

## [v0.13.4] — 2026-04-16

### Added

- add HTML report export with Jinja2 templates (A5)

## [v0.13.3] — 2026-04-16

### Added

- bidirectional citation snowball with budget-aware stopping (A4)

## [v0.13.2] — 2026-04-16

### Added

- evidence-only aggregation (A3)

## [v0.13.1] — 2026-04-16

### Added

- three-channel eval logging (A2)

## [v0.13.0] — 2026-04-16

### Added

- add user feedback loop for screening weight adjustment (v0.13.0)

## [v0.12.14] — 2026-04-16

### Added

- add MCP zero-trust security (T3-6)

## [v0.12.13] — 2026-04-16

### Added

- add multi-agent architecture (T3-5)

## [v0.12.12] — 2026-04-15

### Added

- add tiered page dispatch (T3-4, v0.12.12)

## [v0.12.11] — 2026-04-15

### Added

- add self-improving retrieval (T3-3, v0.12.11)

## [v0.12.10] — 2026-04-15

### Added

- schema-grounded evaluation with per-stage validation

## [v0.12.9] — 2026-04-15

### Added

- content security gates with classification and taint tracking

## [v0.12.8] — 2026-04-15

### Added

- three-tier memory architecture (working/episodic/semantic)

## [v0.12.7] — 2026-04-15

### Added

- THINK→EXECUTE→REFLECT iterative gap-filling loop

## [v0.12.6] — 2026-04-15

### Chore

- bump pyproject.toml version to 0.12.6

## [v0.12.5] — 2026-04-15

### Added

- per-claim confidence scoring with multi-signal aggregation

## [v0.12.4] — 2026-04-15

### Chore

- bump pyproject.toml version to 0.12.4

## [v0.12.3] — 2026-04-15

### Added

- claim decomposition with evidence taxonomy

## [v0.12.2] — 2026-04-15

### Added

- MinerU (magic-pdf) conversion backend

## [v0.12.1] — 2026-04-15

### Added

- cross-encoder passage reranking for chunk retrieval

## [v0.12.0] — 2026-04-15

### Added

- Tier 1 enhancements — FACT verification, export formats, query refinement, structured evidence, enhanced comparison

## [v0.11.0] — 2026-04-15

### Added

- Batch 3 (O-T) — LLM providers, screening judge, summarization, diversity, RACE scoring

## [v0.10.0] — 2026-04-14

### Added

- Batch 2 enhancements (I-N) — safety gate, BFS expansion, sanitization, depth gate, checkpoints, hybrid retrieval

## [v0.9.0] — 2026-04-14

### Added

- v0.9.0 — audit logging, backward-preference citation, Q2D query augmentation, FTS5 index search, bibliography extraction, tool integrity hashing

## [v0.8.1] — 2026-04-14

### Chore

- bump version to 0.8.1

## [v0.8.0] — 2026-04-14

### Added

- add P1-P3 quality improvements (v0.8.0)

## [v0.7.1] — 2026-04-14

### Added

- enhanced report template with confidence levels and structured agent output schemas

## [v0.7.0] — 2026-04-14

### Chore

- bump version to 0.7.0

## [v0.6.0] — 2026-04-14

### Chore

- bump version to 0.6.0

## [v0.5.0] — 2026-04-14

### Chore

- bump version to 0.5.0

## [v0.4.0] — 2026-04-08

### Added

- v0.4.0 — merge diverged fixes + new features

## [v0.3.0] — 2026-04-05

### Documentation

- update all documentation for v0.3.0 release

## [v0.2.0] — 2026-04-05

### Added

- multi-backend PDF conversion with registry pattern

## [v0.1.0] — 2026-04-05

### Chore

- prepare v0.1.0 release for PyPI
