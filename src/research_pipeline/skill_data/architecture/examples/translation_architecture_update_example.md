# Architecture Update: LLM-Agent Document Translation System

> Worked example of the `architecture` skill **`update` mode**, applying the
> accepted decisions from `translation_tech_stack_example.md` (which declared
> `Architecture Update Required? = Yes`) into the architecture. It is an
> **example**, not a recommendation. This is an **update note** — it does not
> overwrite `translation_architecture_example.md`.

## Contents

- [1. Generation Metadata](#1-generation-metadata)
- [2. Source Architecture](#2-source-architecture)
- [3. Update Source Documents](#3-update-source-documents)
- [4. Accepted Decisions Applied](#4-accepted-decisions-applied)
- [5. Sections Requiring Update](#5-sections-requiring-update)
- [6. Architecture Patch Summary](#6-architecture-patch-summary)
- [7. Updated ADRs / Decision Register](#7-updated-adrs--decision-register)
- [8. Updated Handoffs](#8-updated-handoffs)
- [9. Compatibility Check](#9-compatibility-check)
- [10. Remaining Open Questions](#10-remaining-open-questions)
- [11. Update Quality-Gate Self-Check](#11-update-quality-gate-self-check)

---

## 1. Generation Metadata

| Field | Value |
|---|---|
| Source architecture | `translation_architecture_example.md` |
| Source architecture version | 0.6.0 |
| Architecture skill version | 0.7.0 |
| Mode | update |
| Generated at | 2026-06-07 |
| Update sources | `translation_tech_stack_example.md` |
| Overwrites architecture by default? | No (update note only) |
| Architecture Update History row to append | `2026-06-07 · update · §14/§20/§21/§17 · applied selected stack (PostgreSQL + DB-backed queue)` |

## 2. Source Architecture

The translation architecture (v0.6.0) kept its tech stack provisional (§7.1/§7.2)
pending stack mode. Invariants to preserve: the deterministic-spine thesis, the
§23 Experience Architecture intent, the §14 state model, the §12/§13 contracts,
and the §17 trust boundaries / tamper-evident audit.

## 3. Update Source Documents

## Resolved Input Artifacts

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | `translation_architecture_example.md` | High | required |
| architecture_tech_stack | `translation_tech_stack_example.md` | High | accepted update source (`Architecture Update Required? = Yes`, §18) |
| architecture_reconciliation | — | Missing | none yet |
| ux_design | `translation-system-ux-design.md` | — | **not** used as an update source (goes through reconcile) |

> The accepted update source is the tech-stack document. The ux-design is present
> but is **not** applied here — its findings flow through `architecture --mode
> reconcile` first.

## 4. Accepted Decisions Applied

| Decision | Source | Evidence | Affected Architecture Sections | Applied? |
|---|---|---|---|---|
| Storage = PostgreSQL | architecture-tech-stack | §4/§7 selection; §18 Update Required = Yes | §14, §20, §17, §21 | yes |
| Queue = PostgreSQL-backed durable queue | architecture-tech-stack | §8 selection | §9, §18, §21 | yes |
| Provider adapter + redaction at the boundary | architecture-tech-stack | §9/§11 selection | §16, §17 | yes |

## 5. Sections Requiring Update

| Architecture Section | Change Required | Reason | Priority |
|---|---|---|---|
| §14 State, Storage, Data Lifecycle | Firm up storage to PostgreSQL; concurrency + backup notes | Stack selected PostgreSQL | High |
| §20 Deployment Architecture | Add a managed/standalone DB to the deployment topology | PostgreSQL is now a runtime dependency | High |
| §17 Security and Trust Boundaries | Add a DB permission model; keep audit application-enforced + tamper-evident | PostgreSQL selected | Medium |
| §9 Container / Runtime View | Queue is DB-backed (no separate broker container) | DB-backed queue selected | Medium |
| §21 ADRs | Supersede the provisional storage note | Final storage decided | High |

## 6. Architecture Patch Summary

| Patch Area | Old Assumption | New Decision | Patch Summary |
|---|---|---|---|
| Storage | SQLite→PostgreSQL provisional (§7.1) | PostgreSQL | Update §14 concurrency/backup, §20 deployment, §17 permissions |
| Queue | embedded vs broker provisional | PostgreSQL-backed durable queue | Update §9 runtime view + §18 recovery semantics (DB-transactional) |
| Audit store | application-enforced + hash-chain tamper-evident | unchanged | Keep the honest wording — PostgreSQL grants are **not** credited with audit immutability |

## 7. Updated ADRs / Decision Register

| ADR | Title | Old Status | New Status | Supersedes |
|---|---|---|---|---|
| ADR-0002a | Storage = PostgreSQL | Proposed (stack §16) | Accepted | ADR-0002 provisional storage note |
| ADR-0007 | Queue = PostgreSQL-backed durable queue | Proposed (stack §16) | Accepted | — |

## 8. Updated Handoffs

§24.2 Tech-Stack Selection Handoff is now resolved (stack chosen). §27
implementation-planning handoff gains "DB schema + migration" as a first
implementable concern. The §24 security-review handoff gains the new DB
permission model to validate.

## 9. Compatibility Check

| Invariant | Preserved? | Note |
|---|---|---|
| Blueprint thesis preserved | yes | Deterministic spine unchanged |
| Product Experience Direction preserved | yes | No UX intent change |
| State model consistent | yes | Same lifecycle states; storage engine only |
| Interface contracts consistent | yes | No contract shape change |
| Security boundaries not weakened | yes | DB permission model **added** (stronger) |
| Observability still sufficient | yes | Provider-adapter redaction preserved |
| Recommended Next Stages still valid / updated | yes | tech-stack-selection now satisfied |

## 10. Remaining Open Questions

| # | Question | Why It Matters | Resolution Path |
|---|---|---|---|
| 1 | Managed vs self-hosted PostgreSQL? | Affects §20 deployment + ops | Decide at deployment; not architecture-blocking |

## 11. Update Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS | `translation_architecture_example.md` | — |
| Update source found | PASS | tech-stack with Update Required = Yes | — |
| Only accepted decisions applied | PASS | only the stack selections; ux-design not applied here | — |
| Changed sections listed | PASS | §14/§20/§17/§9/§21 | — |
| Unaffected sections preserved | PASS | thesis, state model, contracts unchanged | — |
| ADRs / decision register updated | PASS | ADR-0002a supersedes; ADR-0007 added | — |
| Update history updated | PASS | row prepared (see §1) | — |
| Downstream handoffs still valid | PASS | §24.2 resolved; §27 gains DB schema concern | — |

> This is an update **note**. The architecture design document
> (`translation_architecture_example.md`) is **not** overwritten — applying the
> patch to the main file would require an explicit request, the listed change set,
> an appended Update History row, and recoverability.
