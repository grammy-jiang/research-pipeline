# Architecture Update: <Project Name>

> Skeleton for the `architecture` skill **`update` mode** output. Replace every
> `<…>` placeholder. Co-locate with the architecture as
> `<topic-slug>-architecture-update.md`.
>
> Update **applies accepted decisions** into the architecture. By default it
> produces this **update note** (and optionally a proposed
> `…-architecture-design.updated.md`); it **does not overwrite**
> `…-architecture-design.md`. Apply only accepted decisions, never speculation.
> See `references/architecture-update-guide.md`.

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
| Source architecture | `<filename>` |
| Source architecture version | `<version/hash or unknown>` |
| Architecture skill version | `<from manifest.json version or unknown>` |
| Mode | update |
| Generated at | `<date>` |
| Update sources | `<filenames>` |
| Overwrites architecture by default? | No (update note + optional proposed draft) |
| Architecture Update History row to append | `<date · update · affected sections · note>` |

## 2. Source Architecture

<The architecture being updated, its version, and the invariants that must be
preserved (blueprint thesis, Product Experience Direction, state model, contracts,
security boundaries).>

## 3. Update Source Documents

## Resolved Input Artifacts

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | `<path>` | High | required |
| architecture_tech_stack | `<path or —>` | High / Missing | accepted update source (Update Required = Yes) |
| architecture_reconciliation | `<path or —>` | High / Missing | accepted patch recommendations |
| security_review | `<path or —>` | Missing | not found |

> The accepted update source(s) — in priority order: tech-stack with
> `Architecture Update Required? = Yes`; accepted reconciliation; security-review
> findings; newer blueprint; explicit user decision text. **Not** ux-design
> directly (that goes through `reconcile` first).

## 4. Accepted Decisions Applied

| Decision | Source | Evidence | Affected Architecture Sections | Applied? |
|---|---|---|---|---|
| <decision> | <update source> | <why it is accepted> | <§…> | yes/no |

> Only accepted decisions. Do not apply speculative comments or unaccepted
> suggestions.

## 5. Sections Requiring Update

| Architecture Section | Change Required | Reason | Priority |
|---|---|---|---|
| <§…> | <change> | <reason> | High / Medium / Low |

## 6. Architecture Patch Summary

| Patch Area | Old Assumption | New Decision | Patch Summary |
|---|---|---|---|
| <area> | <old/provisional> | <accepted decision> | <what changes> |

> This describes the *patch*; it does not rewrite the architecture document in
> place (unless an explicit safe-overwrite was requested).

## 7. Updated ADRs / Decision Register

| ADR | Title | Old Status | New Status | Supersedes |
|---|---|---|---|---|
| ADR-00NN | <decision> | Proposed / Accepted | Accepted | <prior ADR or —> |

> Promote accepted ADR candidates from the update source; supersede prior ADRs,
> never silently overwrite them.

## 8. Updated Handoffs

<Which §24 downstream handoffs (tech-stack / ux-design / security-review /
test-design) and §27 implementation-planning handoff notes change as a result of
the applied decisions.>

## 9. Compatibility Check

| Invariant | Preserved? | Note |
|---|---|---|
| Blueprint thesis preserved | yes/no | <…> |
| Product Experience Direction preserved | yes/no | <…> |
| State model consistent | yes/no | <…> |
| Interface contracts consistent | yes/no | <…> |
| Security boundaries not weakened | yes/no | <…> |
| Observability still sufficient | yes/no | <…> |
| Recommended Next Stages still valid / updated | yes/no | <…> |

## 10. Remaining Open Questions

| # | Question | Why It Matters | Resolution Path |
|---|---|---|---|
| 1 | <question> | <impact> | <how/when> |

## 11. Update Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS / FAIL | <finding> | <action> |
| Update source found | PASS / FAIL | <finding> | <action> |
| Only accepted decisions applied | PASS / WARNING / FAIL | no speculation applied | <action> |
| Changed sections listed | PASS / WARNING / FAIL | <finding> | <action> |
| Unaffected sections preserved | PASS / WARNING / FAIL | <finding> | <action> |
| ADRs / decision register updated | PASS / WARNING / FAIL | <finding> | <action> |
| Update history updated | PASS / WARNING / FAIL | row prepared for the architecture | <action> |
| Downstream handoffs still valid | PASS / WARNING / FAIL | <finding> | <action> |

> Status legend: PASS / WARNING / FAIL. The update note never overwrites the
> architecture design by default; an overwrite requires an explicit request, a
> listed change set, an appended Update History row, and recoverability.
