# research-pipeline — Implementation Plan

## 1. Document Control

| Field | Value |
|-------|-------|
| Document | Implementation Plan — research-pipeline |
| Version | 0.17.34 |
| Status | *Active — current state and forward work* |
| Last Updated | 2026-05-17 |
| Source of Truth | This document for plan; [System Design](system-design.md) for architecture; [CHANGELOG](changelog.md) for history |

This is the living implementation plan. Closed-out plans (older project phases) are preserved under [`_archive/`](_archive/).

---

## 2. Scope and Purpose

This plan describes:

- **What is implemented today** — the current production capability surface at v0.17.34.
- **What is open work** — items derived from recent audits, ADRs, and the deep-research design document.
- **What is explicitly out of scope** — to prevent scope creep and to mark items the project has decided not to pursue.

The plan is anchored in versioned decisions:

- [System Design](system-design.md) — the canonical architecture.
- [ADRs](adr/README.md) — the 8 architecture decisions to date.
- [CHANGELOG](changelog.md) — per-version history of what shipped.
- [Audit: Deep-Research Compliance (2026-05-17)](audit-deep-research-compliance-2026-05-17.md) — most recent compliance snapshot against the comprehensive deep-research design document.

---

## 3. Current State (v0.17.34)

### 3.1 Capability surface (verified by the 2026-05-17 audit, 924 clauses satisfied)

| Area | Status | Reference |
|------|--------|-----------|
| Academic pipeline — 7 stages (plan → search → screen → download → convert → extract → summarize) | shipped | [System Design §3](system-design.md) |
| Multi-source discovery — arXiv, Google Scholar, Semantic Scholar, OpenAlex, DBLP | shipped | [API Reference](api-reference.md) |
| Cross-source dedup (arXiv ID, DOI, normalized title) | shipped | [System Design](system-design.md) |
| Semantic re-ranking (optional SPECTER2) | shipped | [System Design](system-design.md) |
| Quality evaluation — composite scoring (citations, venue, h-index, recency, reproducibility) | shipped | [System Design](system-design.md) |
| PDF conversion — 3 local + 5 cloud backends with multi-account rotation and cross-service fallback | shipped | [ADR-004](adr/ADR-004-conversion-backend-registry.md) |
| Daily AI intelligence briefing pipeline | shipped | [User Manual](user-manual.md) |
| MCP server — 64 tools, 21 resources, 6 prompts | shipped | [ADR-006](adr/ADR-006-fastmcp-server.md), [API Reference](api-reference.md) |
| Multi-tier memory (working, episodic, semantic KG, CBR) | shipped | [ADR-007](adr/ADR-007-multi-tier-memory.md) |
| 4-layer confidence architecture (L1 fast → L2 adaptive → L3 DINCO → L4 verification) | shipped | [System Design](system-design.md) |
| Hard constraints HC1–HC6 enforced by harness | shipped | [Security Model](security-model.md), [AGENTS.md](../AGENTS.md) |
| Bundled AI skill + sub-agents (Claude Code, Codex CLI, GitHub Copilot CLI) | shipped | [AGENTS.md §AI skill & sub-agents](../AGENTS.md) |

### 3.2 CI and gates

| Gate | Status |
|------|--------|
| `make verify` — lint + typecheck + test + security | passing |
| `uv run pre-commit run --all-files` | passing |
| `detect-secrets` baseline | clean |
| `mypy --strict` | zero errors enforced |
| Test coverage (unit) | ≥85 % core modules ≥95 % |
| Compliance vs. comprehensive design (`audit` skill) | 924 satisfied, 0 violated, 29 unknown (tooling-coverage gap) |

---

## 4. Open Work

### 4.1 Audit-driven items (from `audit-deep-research-compliance-2026-05-17.md` §6)

| # | Item | Class | Owner |
|---|------|-------|-------|
| OW-1 | Re-audit once upstream `llm-sca-tooling` restores `get_relevant_files` and the per-clause resource handlers; re-classify the 29 unknowns | tooling-blocked | auditor |
| OW-2 | Optional `tests/contracts/` runtime-test layer that materialises unknown-clause runtime assertions (§6 MCP server, §11 evaluation gates of the deep-research design) | testing | tech lead |
| OW-3 | File upstream issues for the four tooling gaps documented in [`upstream-tooling-issues.md`](upstream-tooling-issues.md) | tooling | maintainer |
| OW-4 | Resolve any remaining doc cross-reference gaps once this plan is published | docs | doc owner |

### 4.2 Design-document follow-through

The comprehensive deep-research design document
(`/home/grammy-jiang/Documents/Research/deep-research/deep-research-system-architecture-design.md`)
is the source for forward-looking design. Phased implementation per the
design's Capability Scope Matrix:

| Design phase | Status in research-pipeline |
|--------------|------------------------------|
| P0 (project skeleton) | shipped pre-v0.1 |
| P1 (discovery pipeline) | shipped |
| P2 (analysis data-prep) | shipped |
| P3 (MCP server + intelligence integration) | shipped (v0.5+) |
| P4 (multi-source expansion + memory) | partial: memory shipped (v0.10+); web/GitHub/patent sources scaffolded only |
| P5 (quality & security) | partial: composite quality shipped; epistemic blinding deferred |
| Future (extended MCP + remote access) | not started |

The capabilities flagged `[scaffold]` in the design's Capability Scope
Matrix (web/GitHub/patent search sources via the `SearchSource` protocol)
have interface stubs in `src/research_pipeline/sources/` but no live
implementation.

### 4.3 Documentation

- The `weak_docs_spec_links: ['implementation plan links missing']`
  finding from the 2026-05-17 readiness audit is addressed by this
  document and the cross-reference fixes in `AGENTS.md` and
  `docs/index.md` (commit alongside this plan).
- Continued ADR cadence for any major decision (the project has 8
  ADRs at v0.17.34; the convention is one new ADR per major architectural
  choice).

---

## 5. Explicitly Out of Scope

- **Hosted SaaS / multi-tenant version** — the project is a local CLI/MCP toolkit; hosting is not on the roadmap.
- **Non-academic deep-research domains** — the design is academic-paper-shaped; generalisation to legal/medical/patent corpora is a separate project.
- **Replacing the LLM-agnostic capability tier with a fixed model** — the LLM-agnostic abstraction is load-bearing (per the deep-research design §1) and stays.
- **Removing the SKILL.md / sub-agent boundary** — the boundary between deterministic code and LLM-driven analysis is intentional and stays.

---

## 6. Cadence and Maintenance

- Update this plan whenever a P-phase item ships (move the row up to §3.1) or whenever a new design item lands in `docs/_archive/` or `adr/`.
- Each new audit run should append a row to §4.1 or close one (mark the OW-x as `done` and link to the closing commit/PR).
- Closed-out plan sections (when a milestone fully ships) move to `_archive/` with a date prefix, mirroring the convention used for the v0.3.0 plan now at `_archive/implementation-plan.md`.

---

## 7. Reference

- [Archived plan — Apr 2026, v0.3.0 baseline](_archive/implementation-plan.md)
- [Archived plan — Daily AI intelligence](_archive/daily-ai-intelligence-implementation-plan.md)
- [System Design](system-design.md) — current architecture
- [ADR index](adr/README.md) — 8 ADRs
- [Audit reports](.) — date-prefixed compliance snapshots (e.g. `audit-deep-research-compliance-2026-05-17.md`)
- [AGENTS.md](../AGENTS.md) — agent-facing contract
- [CHANGELOG](changelog.md) — versioned history
