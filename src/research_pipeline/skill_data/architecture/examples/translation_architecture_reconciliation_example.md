# Architecture Reconciliation: LLM-Agent Document Translation System

> Worked example of the `architecture` skill **`reconcile` mode**, consuming the
> §21 Architecture Feedback from `translation-system-ux-design.md` and mapping it
> onto the architecture. It is an **example**, not a recommendation. Reconcile
> **does not patch** the architecture — it recommends, and `update` mode applies
> accepted changes.

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
| Source architecture | `translation_architecture_example.md` |
| Source architecture version | 0.6.0 |
| Architecture skill version | 0.7.0 |
| Mode | reconcile |
| Generated at | 2026-06-07 |
| Feedback sources | `translation-system-ux-design.md` |
| Patches architecture by default? | No (reconciliation note → handoff to update) |
| Architecture Update Required? | Yes |

## 2. Source Architecture

The translation architecture (v0.6.0). Downstream findings must respect its
invariants: the deterministic-spine thesis, the §23 Experience Architecture
intent, the §14 state model, and the §12/§13 contracts. The ux-design's feedback
items are gaps/enhancements, not blueprint conflicts.

## 3. Feedback Documents Consumed

## Resolved Input Artifacts

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | `translation_architecture_example.md` | High | required |
| ux_design | `translation-system-ux-design.md` | High | primary; has `Architecture Feedback / Required Architecture Updates` (§21) |
| security_review | — | Missing | not found |
| test_design | — | Missing | not found |
| implementation_plan | — | Missing | not found |

## 4. Conflict Summary

| Finding | Source Artifact | Severity | Architecture Gap | Recommended Change | Requires Update Mode? |
|---|---|---|---|---|---|
| Per-segment progress UX | ux-design §21 | Warning | §16 emits only job-level progress | Add a `segment_progressed` observability event | yes |
| Review-reason not user-visible | ux-design §21 | Warning | §14 has a `probe-unavailable` condition flag but no user-facing review-reason field on the review artifact | Add a `review_reason` field to the review-artifact schema | yes |
| Web review console deferred | ux-design §21 | Polish | No Web surface in MVP | Note a post-MVP Web review surface; no MVP change | no |

> These are **enhancements/gaps**, not conflicts with the blueprint — the
> deterministic-spine thesis and UX intent are preserved. None contradicts the
> blueprint.

## 5. Missing Architecture Support

| UX / Test / Security Need | Missing Architecture Support | Affected Section | Suggested Fix |
|---|---|---|---|
| Operator sees per-segment progress | No segment-level progress event | §16 Observability, §23.4 Feedback | Add `segment_progressed` event emitted by the worker per segment |
| Reviewer/operator sees *why* review was required | No review-reason field on the review artifact | §13 Data Contracts, §23.6 Human Review | Add `review_reason` (e.g. low_confidence / probe_unavailable) to the review artifact |

## 6. UX / Test / Security Impact Analysis

Both Warning findings are additive: `segment_progressed` extends the §16 event
catalogue without changing the state model; `review_reason` extends the
review-artifact data contract (§13) and the §23.6 human-review flow without
changing lifecycle states. Neither weakens security or egress. The Polish item is
a future surface decision, out of MVP scope.

## 7. Recommended Architecture Changes

| # | Change | Affected Section | Severity | Rationale (minimal?) |
|---|---|---|---|---|
| 1 | Add `segment_progressed` observability event | §16 | Warning | Smallest change: one event; progress UX maps to it |
| 2 | Add `review_reason` field to the review artifact | §13, §23.6 | Warning | One field; surfaces the existing `probe-unavailable` flag to the user |
| 3 | Record a post-MVP Web review surface as deferred | §24 / roadmap | Polish | No MVP architecture change |

## 8. Minimal Patch Plan

Apply two additive changes: (1) extend the §16 event catalogue with
`segment_progressed`; (2) extend the §13 review-artifact contract and §23.6 flow
with a `review_reason` field. Both are non-breaking, preserve all invariants, and
unblock the ux-design's progress and review-reason UX. The Web console stays a
roadmap note.

## 9. Architecture Update Required?

| Update Required | Reason | Recommended Next Command |
|---|---|---|
| Yes | Two accepted additive changes (segment progress event; review-reason field) close the ux-design gaps | architecture --mode update |

## 10. Handoff to architecture --mode update

`architecture --mode update` should consume this reconciliation document. The two
Warning recommendations (`segment_progressed` event, `review_reason` field) are
**accepted** and become an update source; the Polish Web-console item is **open**
(roadmap, no update needed now). No blueprint conflict requires a user decision.

## 11. Reconciliation Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS | `translation_architecture_example.md` | — |
| Feedback document found | PASS | ux-design with §21 Architecture Feedback | — |
| Findings traceable to downstream artifacts | PASS | every §4 row cites ux-design §21 | — |
| Conflicts separated from enhancements | PASS | all items are additive gaps; no blueprint conflict | — |
| Recommended changes are minimal | PASS | one event + one field; no redesign | — |
| Architecture update requirement explicit | PASS | §9 = Yes | — |
| Downstream artifact not blindly accepted | PASS | confirmed no blueprint/PED contradiction before recommending | — |

> Reconcile produced a note and a handoff; it did **not** patch
> `translation_architecture_example.md`. `architecture --mode update` applies the
> two accepted additive changes.
