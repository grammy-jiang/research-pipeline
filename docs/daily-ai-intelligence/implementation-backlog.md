# Daily AI Intelligence — Implementation Backlog

## Phase A

Implement tickets strictly in order unless a previous ticket is reopened.

### A01 — Add briefing package skeleton and models

Target: Create `src/research_pipeline/briefing/` and core Phase A domain models.

### A02 — Add source registry loader and validation

Target: Load and validate Phase A source registry configuration.

### A03 — Add GitHub releases adapter with fixtures

Target: Convert registry-allowed GitHub release data into `IntelligenceEvent` records using offline fixtures.

### A04 — Add RSS/Atom adapter with fixtures

Target: Convert conservative RSS/Atom fields into `IntelligenceEvent` records using offline fixtures.

### A05 — Add normalization and stable ID generation

Target: Implement deterministic normalization and stable IDs for events, content hashes, dedup keys, and cluster IDs.

### A06 — Add exact deduplication

Target: Deduplicate normalized events by exact deterministic keys.

### A07 — Add deterministic ranking and tie-breakers

Target: Rank clusters using Phase A deterministic scoring and stable tie-breakers.

### A08 — Add Markdown daily report renderer

Target: Render a template/extractive daily Markdown report from ranked clusters, including low-signal and no-news variants.

### A09 — Add report/event validator

Target: Validate events and reports deterministically.

### A10 — Add CLI command group and fixed artifact layout

Target: Add `research-pipeline brief ...` commands and the fixed Phase A workspace artifact layout.

### A11 — Add telemetry JSONL

Target: Emit append-only JSONL telemetry for Phase A operations.

### A12 — Add offline end-to-end tests for normal, low-signal, and no-news days

Target: Prove Phase A works offline across normal, low-signal, no-news, duplicate, and validation-failure scenarios.

## Later Phases

Phase B-G are intentionally blocked until Phase A is complete and audited.
