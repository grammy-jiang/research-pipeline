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
- [5. Patch Manifest](#5-patch-manifest)
- [6. Sections Requiring Update](#6-sections-requiring-update)
- [7. Architecture Patch Summary](#7-architecture-patch-summary)
- [8. Updated ADRs / Decision Register](#8-updated-adrs--decision-register)
- [9. Updated Handoffs](#9-updated-handoffs)
- [10. Compatibility Check](#10-compatibility-check)
- [11. Remaining Open Questions](#11-remaining-open-questions)
- [12. Feedback Closure Matrix](#12-feedback-closure-matrix)
- [13. Update Quality-Gate Self-Check](#13-update-quality-gate-self-check)

---

## 1. Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | architecture_update |
| Topic Slug | `<stable-pipeline-slug>` |
| Project Name | `<Project Name>` |
| Skill Name | architecture |
| Skill Version | `<from manifest.json version or UNKNOWN — resolver could not determine this value>` |
| Source architecture | `<filename>` |
| Source architecture version | `<version/hash or UNKNOWN — resolver could not determine this value>` |
| Mode | update |
| Generated at | `<date>` |
| Update sources | `<filenames>` |
| Patch Type | `<NONE / NOTE_ONLY / ADR_ONLY / CONTRACT_PATCH / SECURITY_PATCH / OBSERVABILITY_PATCH / STRUCTURAL_PATCH / BREAKING_CHANGE>` |
| Overwrites architecture by default? | No (update note + optional proposed draft) |
| Architecture Update History row to append | `<date · update · affected sections · note>` |

## Cross-Skill Artifact Contract

> Conforms to the Cross-Skill Artifact Contract
> (`references/artifact-contract.md`).

### Source Artifacts Consumed

| Artifact Role | Path | Required? | How Used |
|---|---|---:|---|
| architecture_design | `<path>` | yes | The architecture being updated |
| architecture_tech_stack | `<path or —>` | no | Accepted update source (Update Required = Yes) |
| architecture_reconciliation | `<path or —>` | no | Accepted patch recommendations |

### Contract Field Map

| Contract Field | Where in this document |
|---|---|
| Generation Metadata | §1 |
| Resolved Input Artifacts | §3 Update Source Documents |
| Decision Register | §4 Accepted Decisions Applied |
| Assumptions | §9 Compatibility Check (invariants) |
| Open Questions | §10 Remaining Open Questions |
| Recommended Next Stage | §8 Updated Handoffs (re-review / implementation-plan) |
| Quality-Gate Self-Check | §11 (incl. the Cross-Skill Artifact Contract Gate) |

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

## 5. Patch Manifest

> Machine-readable patch description for `architecture --mode materialize`.
> See `references/patch-manifest-guide.md` for the full format guide.

```yaml
patch_manifest:
  - id: <patch-id>                      # e.g. RC-1, STACK-2, SEC-3
    source: <source artifact filename>
    target_section: "<section number>"  # e.g. "12.1", "16.2", "17"
    operation: <operation>              # append_subsection | append_row | replace_subsection | update_row | update_adrs | add_note | remove_provisional
    patch_type: <TYPE>                  # from taxonomy: NOTE_ONLY | ADR_ONLY | CONTRACT_PATCH | SECURITY_PATCH | OBSERVABILITY_PATCH | STRUCTURAL_PATCH | BREAKING_CHANGE
    blocks_implementation: true/false
    description: "<short description of what changes>"
```

> If this update was produced without an explicit patch manifest, emit
> `NOT_APPLICABLE — patch manifest not available; materialize will infer from §6 and §7`.
> The materialization quality gate will warn.

## 6. Sections Requiring Update

| Architecture Section | Change Required | Reason | Priority |
|---|---|---|---|
| <§…> | <change> | <reason> | High / Medium / Low |

## 7. Architecture Patch Summary

| Patch Area | Old Assumption | New Decision | Patch Summary |
|---|---|---|---|
| <area> | <old/provisional> | <accepted decision> | <what changes> |

> This describes the *patch*; it does not rewrite the architecture document in
> place (unless an explicit safe-overwrite was requested).

## 8. Updated ADRs / Decision Register

| ADR | Title | Old Status | New Status | Supersedes |
|---|---|---|---|---|
| ADR-00NN | <decision> | Proposed / Accepted | Accepted | <prior ADR or —> |

> Promote accepted ADR candidates from the update source; supersede prior ADRs,
> never silently overwrite them.

## 9. Updated Handoffs

<Which §24 downstream handoffs (tech-stack / ux-design / security-review /
test-design) and §27 implementation-planning handoff notes change as a result of
the applied decisions.>

## 10. Compatibility Check

| Invariant | Preserved? | Note |
|---|---|---|
| Blueprint thesis preserved | yes/no | <…> |
| Product Experience Direction preserved | yes/no | <…> |
| State model consistent | yes/no | <…> |
| Interface contracts consistent | yes/no | <…> |
| Security boundaries not weakened | yes/no | <…> |
| Observability still sufficient | yes/no | <…> |
| Recommended Next Stages still valid / updated | yes/no | <…> |

## 11. Remaining Open Questions

| ID | Question | Source | Owner Stage | Blocks | Status | Required Resolution |
|---|---|---|---|---|---|---|
| OQ-1 | <question> | <source artifact> | implementation-plan | <milestone> | OPEN | <action> |

## 12. Feedback Closure Matrix

> Include this section when this update applies feedback from a downstream
> artifact (e.g. ux-design Architecture Feedback, security-review findings).
> If not applicable, write `NOT_APPLICABLE — this update does not apply
> downstream feedback`.

| Feedback Item | Source | Closed By | Status |
|---|---|---|---|
| <feedback-id> | <source artifact §N> | <patch-id(s)> | RESOLVED / PARTIAL / OPEN |

## 13. Update Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS / FAIL | <finding> | <action> |
| Update source found | PASS / FAIL | <finding> | <action> |
| Only accepted decisions applied | PASS / WARNING / FAIL | no speculation applied | <action> |
| Patch manifest present | PASS / WARNING / FAIL | <present or inferred> | <action> |
| Changed sections listed | PASS / WARNING / FAIL | <finding> | <action> |
| Unaffected sections preserved | PASS / WARNING / FAIL | <finding> | <action> |
| ADRs / decision register updated | PASS / WARNING / FAIL | <finding> | <action> |
| Update history updated | PASS / WARNING / FAIL | row prepared for the architecture | <action> |
| Downstream handoffs still valid | PASS / WARNING / FAIL | <finding> | <action> |
| Feedback closure matrix present (when applicable) | PASS / WARNING / FAIL / NOT_APPLICABLE | <finding> | <action> |
| Skill version metadata known | PASS / WARNING / FAIL | <UNKNOWN fields warn> | <action> |

> Status legend: PASS / WARNING / FAIL. The update note never overwrites the
> architecture design by default; an overwrite requires an explicit request, a
> listed change set, an appended Update History row, and recoverability.

### Cross-Skill Artifact Contract Gate

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Generation metadata present (Artifact Type + Topic Slug) | PASS / WARNING / FAIL | <finding> | <action> |
| Topic slug present and stable | PASS / WARNING / FAIL | <finding> | <action> |
| Source artifacts listed | PASS / WARNING / FAIL | <finding> | <action> |
| Resolved input artifacts recorded (when discovery is used) | PASS / WARNING / FAIL / NOT_APPLICABLE | <finding> | <action> |
| Decisions and assumptions separated | PASS / WARNING / FAIL | §4 decisions vs §10 invariants | <action> |
| Open questions assigned to a next stage | PASS / WARNING / FAIL | §11 | <action> |
| Recommended next stage present | PASS / WARNING / FAIL | §9 | <action> |

> See `references/artifact-contract.md`.
