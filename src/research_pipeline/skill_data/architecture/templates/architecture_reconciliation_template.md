# Architecture Reconciliation: <Project Name>

> Skeleton for the `architecture` skill **`reconcile` mode** output. Replace
> every `<…>` placeholder. Co-locate with the architecture as
> `<topic-slug>-architecture-reconciliation.md`.
>
> Reconcile **detects gaps/conflicts between the architecture and downstream
> artifacts** (ux-design / test-design / security-review / implementation-plan).
> It **does not patch the architecture by default** — it produces findings and
> recommended changes, then hands off accepted ones to `update` mode. See
> `references/architecture-reconciliation-guide.md`.

## Contents

- [1. Generation Metadata](#1-generation-metadata)
- [2. Source Architecture](#2-source-architecture)
- [3. Feedback Documents Consumed](#3-feedback-documents-consumed)
- [4. Conflict Summary](#4-conflict-summary)
- [5. Missing Architecture Support](#5-missing-architecture-support)
- [6. UX / Test / Security Impact Analysis](#6-ux--test--security-impact-analysis)
- [7. Recommended Architecture Changes](#7-recommended-architecture-changes)
- [8. Minimal Patch Plan](#8-minimal-patch-plan)
- [9. Architecture Update Required?](#9-architecture-update-required)
- [10. Handoff to architecture --mode update](#10-handoff-to-architecture---mode-update)
- [11. Reconciliation Quality-Gate Self-Check](#11-reconciliation-quality-gate-self-check)

---

## 1. Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | architecture_reconciliation |
| Topic Slug | `<stable-pipeline-slug>` |
| Project Name | `<Project Name>` |
| Skill Name | architecture |
| Source architecture | `<filename>` |
| Source architecture version | `<version/hash or unknown>` |
| Architecture skill version | `<from manifest.json version or unknown>` |
| Mode | reconcile |
| Generated at | `<date>` |
| Feedback sources | `<filenames>` |
| Patches architecture by default? | No (reconciliation note → handoff to update) |
| Architecture Update Required? | Yes / No |

## Cross-Skill Artifact Contract

> Conforms to the Cross-Skill Artifact Contract
> (`references/artifact-contract.md`).

### Source Artifacts Consumed

| Artifact Role | Path | Required? | How Used |
|---|---|---:|---|
| architecture_design | `<path>` | yes | The architecture being reconciled |
| ux_design | `<path or —>` | no | Primary feedback (Architecture Feedback section) |
| security_review | `<path or —>` | no | Architecture-impacting findings |
| test_design | `<path or —>` | no | Impossible / uncovered scenarios |

### Contract Field Map

| Contract Field | Where in this document |
|---|---|
| Generation Metadata | §1 |
| Resolved Input Artifacts | §3 Feedback Documents Consumed |
| Decision Register | §7 Recommended Architecture Changes (proposed; accepted by `update`) |
| Assumptions | §6 UX / Test / Security Impact Analysis |
| Open Questions | §4 Conflict Summary (open conflicts) |
| Recommended Next Stage | §9 Architecture Update Required? + §10 Handoff |
| Quality-Gate Self-Check | §11 (incl. the Cross-Skill Artifact Contract Gate) |

## 2. Source Architecture

<The architecture being reconciled, its version, and the invariants that
downstream findings must respect (blueprint thesis, Product Experience Direction,
state model, contracts).>

## 3. Feedback Documents Consumed

## Resolved Input Artifacts

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | `<path>` | High | required |
| ux_design | `<path or —>` | High / Missing | primary feedback source (Architecture Feedback section) |
| security_review | `<path or —>` | Missing | not found |
| test_design | `<path or —>` | Missing | not found |
| implementation_plan | `<path or —>` | Missing | not found |

## 4. Conflict Summary

| Finding | Source Artifact | Severity | Architecture Gap | Recommended Change | Requires Update Mode? |
|---|---|---|---|---|---|
| <finding> | ux-design / test-design / security-review | Blocking / Warning / Polish | <gap> | <minimal change> | yes/no |

> Separate genuine **conflicts** (downstream needs something the architecture
> contradicts or lacks) from **enhancements** (nice-to-haves). Do not blindly
> accept a downstream artifact that contradicts the blueprint — flag it.

## 5. Missing Architecture Support

| UX / Test / Security Need | Missing Architecture Support | Affected Section | Suggested Fix |
|---|---|---|---|
| <need> | <what the architecture lacks> | <§…> | <minimal fix> |

## 6. UX / Test / Security Impact Analysis

<For each high-severity finding: how it affects the state model, contracts,
observability, security, or surfaces — and whether the downstream artifact or the
architecture should change.>

## 7. Recommended Architecture Changes

| # | Change | Affected Section | Severity | Rationale (minimal?) |
|---|---|---|---|---|
| 1 | <change> | <§…> | Blocking / Warning / Polish | <smallest change that closes the gap> |

## 8. Minimal Patch Plan

<The smallest coherent set of architecture changes that resolves the Blocking +
accepted Warning findings. Group related changes. This is a plan, not a patch —
`update` mode applies the accepted ones.>

## 9. Architecture Update Required?

| Update Required | Reason | Recommended Next Command |
|---|---|---|
| Yes / No | <reason> | architecture --mode update |

## 10. Handoff to architecture --mode update

<If Update Required = Yes: state that `architecture --mode update` should consume
this reconciliation document, and which recommendations are **accepted** (and
thus an update source) vs **open** (need a user decision first). If No: state
that no architecture change is needed and why.>

## 11. Reconciliation Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS / FAIL | <finding> | <action> |
| Feedback document found | PASS / FAIL | <finding> | <action> |
| Findings traceable to downstream artifacts | PASS / WARNING / FAIL | each finding cites its source | <action> |
| Conflicts separated from enhancements | PASS / WARNING / FAIL | <finding> | <action> |
| Recommended changes are minimal | PASS / WARNING / FAIL | <finding> | <action> |
| Architecture update requirement explicit | PASS / WARNING / FAIL | §9 verdict present | <action> |
| Downstream artifact not blindly accepted | PASS / WARNING / FAIL | blueprint conflicts flagged | <action> |

> Status legend: PASS / WARNING / FAIL. Reconcile never patches the architecture
> by default — it recommends, and `update` mode applies the accepted changes.

### Cross-Skill Artifact Contract Gate

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Generation metadata present (Artifact Type + Topic Slug) | PASS / WARNING / FAIL | <finding> | <action> |
| Topic slug present and stable | PASS / WARNING / FAIL | <finding> | <action> |
| Source artifacts listed | PASS / WARNING / FAIL | <finding> | <action> |
| Resolved input artifacts recorded (when discovery is used) | PASS / WARNING / FAIL / NOT_APPLICABLE | <finding> | <action> |
| Decisions and assumptions separated | PASS / WARNING / FAIL | §7 recommendations vs §6 analysis | <action> |
| Open questions assigned to a next stage | PASS / WARNING / FAIL | §4 / §9 | <action> |
| Recommended next stage present | PASS / WARNING / FAIL | §9 Architecture Update Required? | <action> |

> See `references/artifact-contract.md`.
