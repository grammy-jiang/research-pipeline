# Testing Strategy: research-pipeline

## 1. Document Purpose

This document describes the testing philosophy, pyramid, conventions, and
tooling used in `research-pipeline`. It is the authoritative reference for
contributors writing tests and for operators evaluating test coverage.

---

## 2. Testing Philosophy

`research-pipeline` follows **test-first development** for all new features
and **regression tests required for all bug fixes**. The guiding principles are:

1. **Tests are the specification.** Behaviour not covered by a test is not
   guaranteed to exist.
2. **Speed matters.** The unit suite must remain fast (< 60 s) so it runs on
   every commit without friction.
3. **Determinism required.** No test should fail intermittently due to network
   conditions, race conditions, or system state.
4. **Independence.** Every test must be able to run in isolation. Tests must not
   depend on execution order.
5. **Never modify existing tests without approval.** Changing a test to make it
   pass is a code smell — fix the code, not the test.

---

## 3. Test Pyramid

```
          ┌───────────────────┐
          │   Live (network)  │  ← @pytest.mark.live; real arXiv API
          ├───────────────────┤
          │ Integration       │  ← VCR cassettes; offline HTTP replay
          │ (offline)         │
          ├───────────────────┤
          │   Unit (4,378+)   │  ← Fast; no network; runs on every commit
          └───────────────────┘
```

| Layer | Directory | Tests | Dependencies | CI? |
|-------|-----------|-------|--------------|-----|
| Unit | `tests/unit/` | 4,378+ in 226 files | None (no network) | ✅ Always |
| Integration (offline) | `tests/integration_offline/` | ~12 files | VCR cassettes | ✅ Always |
| Live | `tests/live/` | Varies | Real arXiv/Scholar APIs | ❌ Manual only |

---

## 4. Test Tooling

| Tool | Purpose |
|------|---------|
| `pytest` ≥ 8.0 | Test runner and framework |
| `pytest-cov` ≥ 5.0 | Coverage reporting |
| `vcrpy` ≥ 6.0 | HTTP cassette recording/replay |
| `hypothesis` ≥ 6.100 | Property-based testing |
| `pytest-anyio` | Async test support |
| `mypy` | Type-check tests along with source |
| `ruff` | Lint test files |

### Install dev dependencies

```bash
uv sync --extra dev
```

---

## 5. Unit Tests

### Location

```
tests/unit/test_<module>.py
```

Every test file name mirrors the source module it tests:

| Test file | Source module |
|-----------|---------------|
| `test_heuristic.py` | `src/research_pipeline/screening/heuristic.py` |
| `test_briefing_rank.py` | `src/research_pipeline/briefing/rank.py` |
| `test_workspace.py` | `src/research_pipeline/storage/workspace.py` |

### Running unit tests

```bash
# Full unit suite
uv run pytest tests/unit/ -x -q

# Single file
uv run pytest tests/unit/test_heuristic.py -xvs

# Single test
uv run pytest tests/unit/test_heuristic.py::TestBM25Scorer::test_basic_score -xvs

# With coverage
uv run pytest tests/unit/ --cov=src/research_pipeline --cov-report=term-missing
```

> **CI gate**: Coverage must stay ≥ 83% (enforced in CI with `--cov-fail-under=83`).

### Unit test conventions

- Test class names: `Test<ClassUnderTest>` (checked by `name-tests-test` hook)
- Test function names: `test_<behaviour>` or `test_<function>_<condition>`
- Each test should exercise exactly one behaviour or outcome
- Use `pytest.raises` for expected exceptions, not `try/except`
- Never `import research_pipeline` at module level in test files without a
  fixture to handle missing optional dependencies

#### Example unit test pattern

```python
import pytest
from research_pipeline.screening.heuristic import BM25Scorer


class TestBM25Scorer:
    def test_scores_relevant_paper_higher(self) -> None:
        scorer = BM25Scorer(queries=["transformer attention mechanism"])
        candidates = [
            {"title": "Attention Is All You Need", "abstract": "transformer model"},
            {"title": "Linear Regression Tutorial", "abstract": "basic statistics"},
        ]
        scores = scorer.score_batch(candidates)
        assert scores[0] > scores[1]

    def test_empty_corpus_returns_zero_scores(self) -> None:
        scorer = BM25Scorer(queries=["any topic"])
        assert scorer.score_batch([]) == []
```

---

## 6. Integration Tests (Offline)

### Location

```
tests/integration_offline/
```

### Purpose

Test HTTP-dependent code (arXiv API, Semantic Scholar, etc.) without network
access by replaying recorded HTTP interactions via VCR cassettes.

### Cassette files

HTTP cassettes live in `tests/fixtures/http_cassettes/`. They are committed to
the repository and replayed deterministically.

### Running integration tests

```bash
uv run pytest tests/integration_offline/ -xvs
```

### Recording a new cassette

To record a new cassette, run the test with a live network connection:

```python
@pytest.fixture
def vcr_cassette_dir(tmp_path):
    return "tests/fixtures/http_cassettes"

@pytest.mark.vcr("my_cassette.yaml")
def test_arxiv_search():
    # This runs live once to record, then replays
    ...
```

### VCR configuration

Default VCR config records/replays `GET` and `POST` requests. Sensitive
headers (Authorization, API keys) are scrubbed from cassettes before commit.

---

## 7. Live Tests

### Location

```
tests/live/
```

### Purpose

End-to-end tests against real external APIs (arXiv, Semantic Scholar).
These are **not run in CI** and must be run manually by a developer with
network access.

### Marking live tests

```python
import pytest

@pytest.mark.live
def test_live_arxiv_search():
    # Requires real arXiv API access
    ...
```

### Running live tests

```bash
# Run all live tests
uv run pytest tests/live/ -xvs -m live

# Run unit and integration but skip live
uv run pytest tests/unit/ tests/integration_offline/ -m "not live"
```

---

## 8. Test Fixtures

### Location

```
tests/fixtures/
├── atom/          # arXiv Atom XML responses
├── briefing/      # Briefing pipeline fixtures
├── http_cassettes/# VCR HTTP cassettes
├── llm/           # LLM response fixtures
├── markdown/      # Converted Markdown paper fixtures
└── pdf/           # PDF files for conversion tests
```

### Shared fixtures

Module-scope and session-scope shared fixtures are defined in `conftest.py`
files at each test level. Common fixtures include:

- `tmp_workspace` — creates an isolated temporary workspace directory
- `sample_query_plan` — a minimal `QueryPlan` instance
- `sample_candidates` — a list of `CandidateRecord` instances

---

## 9. Property-Based Testing

[Hypothesis](https://hypothesis.readthedocs.io/) is available for property-based
tests. Use it when the range of valid inputs is large or hard to enumerate:

```python
from hypothesis import given, strategies as st
from research_pipeline.infra.hashing import sha256_of_str

@given(st.text())
def test_hash_is_deterministic(text: str) -> None:
    assert sha256_of_str(text) == sha256_of_str(text)
```

---

## 10. Coverage Requirements

| Scope | Minimum coverage |
|-------|-----------------|
| `src/research_pipeline/` (overall) | 83% (CI-enforced) |
| New code in PRs | Should maintain or improve coverage |

> **Note**: `src/research_pipeline/mcp_server/` is excluded from mypy strict mode
> (`ignore_errors = true` in `pyproject.toml`) but is still included in coverage.

Coverage report generation:

```bash
uv run pytest tests/unit/ --cov=src/research_pipeline --cov-report=html
# Open htmlcov/index.html
```

---

## 11. Static Analysis as Tests

The following static analysis tools are run as part of the pre-commit suite
and CI. Failures here are treated as test failures:

| Tool | What it checks |
|------|---------------|
| `mypy` (strict mode) | Type correctness, 0 errors required |
| `ruff check` | Lint rules (A, B, C4, E, F, I, PT, RUF, SIM, UP, W) |
| `ruff format` | Code formatting |
| `bandit` | SAST security rules |
| `detect-secrets` | Secret scanning against `.secrets.baseline` |
| `pip-audit` | Vulnerable dependency detection |
| `pip-licenses` | GPL/AGPL copyleft license detection |

```bash
# Run all at once
uv run pre-commit run --all-files
```

---

## 12. CI Pipeline

All tests and checks run on every push and pull request to `master`.

```yaml
Jobs:
  lint:   pre-commit on all files (Python 3.12)
  test:   pytest tests/unit/ (Python 3.12 + 3.13)
           --cov-fail-under=83
  typecheck: mypy src/ (strict)
  security: pip-audit + pip-licenses (no GPL/AGPL)
```

Coverage is uploaded to Codecov on Python 3.12 runs.

---

## 13. Writing a New Test

### Step-by-step

1. **Identify the module** to test: e.g., `src/research_pipeline/retrieval/hybrid.py`
2. **Create the test file** if it doesn't exist: `tests/unit/test_retrieval_hybrid.py`
3. **Write the test first** (TDD), then implement the feature
4. **Use the `Test<Class>` naming pattern** for classes
5. **Keep each test atomic**: one assertion per test where practical
6. **Mock external dependencies** with `unittest.mock.patch` or `pytest.monkeypatch`
7. **Use fixtures** for shared setup; avoid `setUp`/`tearDown` (prefer `@pytest.fixture`)

### Checklist before committing a test

- [ ] Test file is in `tests/unit/test_<module>.py`
- [ ] Test class name starts with `Test`
- [ ] Test function names start with `test_`
- [ ] No `print()` statements in tests
- [ ] No hardcoded file paths that wouldn't work on CI
- [ ] No network calls (add `@pytest.mark.live` if unavoidable)
- [ ] Test passes in isolation: `uv run pytest tests/unit/test_mymodule.py -xvs`
- [ ] Pre-commit passes: `uv run pre-commit run --all-files`

---

## 14. Test Maintenance

### When to update existing tests

Never modify existing tests **except** for:

1. Fixing a test that was factually wrong (not just failing)
2. Updating test fixture data when a schema changes (with explicit approval)
3. Renaming symbols that were refactored across the codebase

### Flaky test policy

If a test is intermittently failing due to timing or non-determinism:

1. Open an issue immediately
2. Mark as `@pytest.mark.xfail(strict=False, reason="flaky: <issue-link>")` temporarily
3. Fix within the next release cycle

### When tests find bugs in production code

Follow the test-driven bug repair cycle:

1. Write a failing test that reproduces the bug (do NOT fix code yet)
2. Verify the test fails on `main`
3. Fix the production code
4. Verify the test passes
5. Submit both test and fix together in the same commit/PR

---

## 15. Appendix: Useful pytest Flags

| Flag | Purpose |
|------|---------|
| `-x` | Stop at first failure |
| `-v` | Verbose output |
| `-s` | Disable output capture (show print/logging) |
| `--tb=short` | Short tracebacks |
| `--tb=no` | No tracebacks (for collection-only checks) |
| `-k "keyword"` | Run tests matching keyword |
| `--co` | Collect only (no execution) |
| `-m "not live"` | Skip live-marked tests |
| `--lf` | Re-run last failed tests |
| `--pdb` | Drop to debugger on failure |
| `-n auto` | Parallel execution (requires pytest-xdist) |
