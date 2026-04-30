# Proof Pack — A12: Offline E2E Tests (Normal / Low-Signal / No-News)

## Verification Status

**PASS** — 29/29 integration tests pass, ruff clean, mypy clean.

## Verification Commands

```bash
cd /home/grammy-jiang/projects/research-pipeline

uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py -v
# 29 passed in 0.51s

uv run ruff check tests/integration_offline/test_briefing_phase_a_e2e.py
# All checks passed!

uv run mypy src/research_pipeline/briefing tests/integration_offline/test_briefing_phase_a_e2e.py
# Success: no issues found in 29 source files
```

## Test Coverage

| Class | Count | Description |
|---|---|---|
| `TestNormalDay` | 12 | Full pipeline on normal day with 2 sources |
| `TestLowSignalDay` | 4 | Pipeline with minimal events (1 release, empty RSS) |
| `TestNoNewsDay` | 6 | Pipeline with zero events from all sources |
| `TestDuplicateDeduplication` | 2 | Canonical URL dedup across clusters |
| `TestArtifactLayout` | 5 | Fixed layout paths + parametrized 3-scenario check |
| **Total** | **29** | |

## Owned Files

### Test file
- `tests/integration_offline/test_briefing_phase_a_e2e.py`

### Fixtures
- `tests/fixtures/briefing/e2e/normal/registry.toml` — 2 sources, watchlist `["MCP", "agent"]`
- `tests/fixtures/briefing/e2e/normal/github_releases.json` — 2 releases
- `tests/fixtures/briefing/e2e/normal/feed.xml` — 2 RSS items
- `tests/fixtures/briefing/e2e/low_signal/registry.toml` — 2 sources, watchlist `["MCP"]`
- `tests/fixtures/briefing/e2e/low_signal/github_releases.json` — 1 release
- `tests/fixtures/briefing/e2e/low_signal/feed.xml` — empty RSS channel
- `tests/fixtures/briefing/e2e/no_news/registry.toml` — 2 sources
- `tests/fixtures/briefing/e2e/no_news/github_releases.json` — empty list `[]`
- `tests/fixtures/briefing/e2e/no_news/feed.xml` — empty RSS channel

## Issues Resolved

- `source_class = "primary_source"` is not a valid `SourceClass` enum value; replaced
  with `"primary_artifact"` in all 6 registry.toml files (3 scenarios × 2 sources).
- Removed unused import `resolve_briefing_paths` from test file (ruff F401).
