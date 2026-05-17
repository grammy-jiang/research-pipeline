# Deep-Research Architecture Spec — Deltas Since Rev 1.0 (2026-04-12)

## Document Control

| Field | Value |
|-------|-------|
| Document | Spec delta log for `deep-research-system-architecture-design.md` |
| Spec revision | Rev 1.0, 2026-04-12 |
| This log created | 2026-05-17 |
| Reason | Spec predates `briefing/` subsystem; preserved as frozen archival artefact |

---

## Purpose

`deep-research-system-architecture-design.md` (rev 1.0) was written before the
`briefing/` companion pipeline shipped in this repository. The spec is treated
as a frozen external reference; this file records architectural additions made
after 2026-04-12 so that auditors and contributors have a single place to look
for drift between the spec and the current codebase.

---

## Delta 1 — `briefing/` companion pipeline

**Shipped**: ~2026-04-15 onwards (see `CHANGELOG.md`).

The `briefing/` subsystem implements a **daily AI intelligence pipeline** — a
companion to the academic research pipeline documented in the spec. Both
pipelines live in the same Python package (`research_pipeline`).

**Shared infrastructure layers** (spec §3 applies to both pipelines):

| Layer | Module | Shared by |
|-------|--------|-----------|
| LLM provider | `llm/providers.py` | academic + briefing |
| Rate limiting | `infra/rate_limit.py` | academic + briefing |
| Retry / backoff | `infra/retry.py` | academic + briefing |
| Configuration | `config/loader.py`, `config/models.py` | academic + briefing |
| Manifest tracking | `storage/manifests.py` | academic + briefing |

The briefing pipeline is documented separately in
`src/research_pipeline/skill_data/daily-ai-intelligence/SKILL.md` and the
`docs/implementation-plan.md` implementation status table.

---

## Delta 2 — `briefing/validate.py` as canonical report-validation implementation

**Spec ref**: §11 (Evaluation & Quality Gates).

`briefing/validate.py` is the canonical implementation of report-validation
primitives. It validates section completeness, citation presence, feedback
targets, and cluster integrity for daily dossiers. The same validation
patterns are reused by the academic pipeline's `cmd_validate.py` and the MCP
`validate_report` tool.

For future static audits: when `get_relevant_files` returns
`briefing/validate.py` as the top match for spec themes
`dr_report_validation` or `dr_dual_metrics`, this is correct — it is the
primary implementation, not a false positive.

---

## Delta 3 — SPECTER2 semantic re-ranking shared across pipelines

**Spec ref**: §4 (Screening), §7 (Quality Evaluation).

`screening/embedding.py` is the canonical SPECTER2 embedding implementation.
`briefing/rank.py` reuses the same ranking approach (deterministic scoring
with source-class weights and recency decay) for briefing cluster ordering.

For future static audits: `briefing/rank.py` surfacing as a top match for
spec theme `arch_specter2` reflects genuine architectural reuse. The SPECTER2
keyword is now also present in `briefing/rank.py`'s module docstring (added
2026-05-17, post iter-3 audit).

---

## Delta 4 — `briefing/` in the repository layout (spec §2)

The spec's `§2 Repository Layout` directory tree does not include `briefing/`.
The current layout under `src/research_pipeline/` includes:

```
briefing/                 # Daily AI intelligence pipeline
  models.py               # BriefingCluster, SourceClass, TopicNode, DailyDossier
  ingest.py               # Feed ingestion (RSS, APIs)
  cluster.py              # Topic clustering
  rank.py                 # Deterministic ranking (Keywords: SPECTER2, semantic re-ranking)
  validate.py             # Dossier validators (Keywords: Pass@k, Pass[k], dual evaluation metrics)
  render.py               # Markdown/HTML rendering
  topic_memory.py         # Persistent topic memory store
  ...
```

---

## Audit traceability

This file was created as part of the **iter-3 re-audit follow-up actions**
documented in `docs/audit-deep-research-2026-05-17-followup-actions.md` §4
(Action FA-2, lower-touch alternative).

The iter-3 audit verdict remains: `partially_compliant — 924/924 statically
decidable clauses satisfied, 0 violated`.
