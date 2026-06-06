# Architecture Review: <Project Name>

> Skeleton for the `architecture` skill **`review` mode** output. Replace every
> `<…>` placeholder. Co-locate with the reviewed architecture as
> `<topic-slug>-architecture-review.md`.
>
> Review **evaluates quality without changing the architecture**. It never
> rewrites, patches, or overwrites the architecture design. See
> `references/architecture-review-guide.md`.

## Contents

- [1. Review Metadata](#1-review-metadata)
- [2. Documents Reviewed](#2-documents-reviewed)
- [3. Executive Assessment](#3-executive-assessment)
- [4. Score Breakdown](#4-score-breakdown)
- [5. Blueprint Fidelity](#5-blueprint-fidelity)
- [6. Product Experience Direction Preservation](#6-product-experience-direction-preservation)
- [7. Recommended Next Stages Consumption](#7-recommended-next-stages-consumption)
- [8. System Architecture Quality](#8-system-architecture-quality)
- [9. State / Contract / Data Model Quality](#9-state--contract--data-model-quality)
- [10. Security / Privacy / Data Egress Review](#10-security--privacy--data-egress-review)
- [11. Observability / Audit Review](#11-observability--audit-review)
- [12. Tech Stack Consistency](#12-tech-stack-consistency)
- [13. UX-Design Readiness](#13-ux-design-readiness)
- [14. Implementation-Plan Readiness](#14-implementation-plan-readiness)
- [15. Blocking Issues](#15-blocking-issues)
- [16. Warnings](#16-warnings)
- [17. Polish Improvements](#17-polish-improvements)
- [18. Recommended Next Actions](#18-recommended-next-actions)
- [19. Review Quality-Gate Self-Check](#19-review-quality-gate-self-check)

---

## 1. Review Metadata

| Field | Value |
|---|---|
| Reviewed architecture | `<filename>` |
| Architecture version | `<version/hash or unknown>` |
| Architecture skill version | `<from manifest.json version or unknown>` |
| Mode | review |
| Generated at | `<date>` |
| Overall score | `<X.X / 10>` |
| Implementation-plan ready? | yes / no / with-changes |

> Review is **non-mutating** — it does not edit the architecture document.

## 2. Documents Reviewed

`## Resolved Input Artifacts` (from the shared resolver — exactly what was read):

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | `<path>` | High | required; matched slug + title |
| product_blueprint | `<path or —>` | Medium / Missing | <reason> |
| architecture_tech_stack | `<path or —>` | High / Missing | <reason> |
| ux_design | `<path or —>` | High / Missing | <reason> |
| security_review | `<path or —>` | Missing | not found |
| test_design | `<path or —>` | Missing | not found |

> Only score content actually present above. Missing optional docs do not fail
> review; note them and scope the score accordingly.

## 3. Executive Assessment

<2–5 sentences: overall quality, the single biggest strength, the single biggest
risk, and whether the architecture is ready for implementation planning.>

## 4. Score Breakdown

| Area | Score | Comment |
|---|---:|---|
| Blueprint fidelity | <0–10> | <justification> |
| Product Experience Direction preservation | <0–10> | <…> |
| Recommended Next Stages consumption | <0–10> | <…> |
| Experience Architecture quality | <0–10> | <…> |
| System decomposition | <0–10> | <…> |
| State model | <0–10> | <…> |
| Interface contracts | <0–10> | <…> |
| Data contracts | <0–10> | <…> |
| Security / trust boundaries | <0–10> | <…> |
| Data egress / privacy | <0–10> | <…> |
| Observability / audit | <0–10> | <…> |
| Failure / recovery | <0–10> | <…> |
| Tech stack separation | <0–10> | <…> |
| UX-design readiness | <0–10> | <…> |
| Implementation-plan readiness | <0–10> | <…> |

> Every score carries a reason. **Overall score = <X.X / 10>.**

## 5. Blueprint Fidelity

<Does the architecture preserve the blueprint thesis, MVP-0/MVP-1, and scope? Cite
specific architecture vs blueprint sections.>

## 6. Product Experience Direction Preservation

<Is the blueprint §9 / architecture §23 Experience Architecture preserved and
sufficient? (n/a if no blueprint — say so.)>

## 7. Recommended Next Stages Consumption

<Does the architecture reflect the blueprint §19 routing (§24 handoffs, provisional
tech when stack recommended)?>

## 8. System Architecture Quality

<C4 coverage, decomposition, container/component ownership, AI boundary, MCP
decision.>

## 9. State / Contract / Data Model Quality

<State model coherence (lifecycle vs condition flags vs audit events), interface
contracts, data contracts + retention/immutability.>

## 10. Security / Privacy / Data Egress Review

<Trust boundaries, data-egress decision, secrets, prompt-injection controls,
security gates as verification tables.>

## 11. Observability / Audit Review

<Correlation IDs, audit trail, log/metric/trace coverage, raw-source logging
policy.>

## 12. Tech Stack Consistency

<Is the stack consistent with the AI boundary + MCP? Is it appropriately
provisional or finalized? (n/a if no stack document — say so.)>

## 13. UX-Design Readiness

<Are surfaces, states, review artifacts, and handoffs sufficient for ux-design?
If a ux-design exists, did the architecture absorb its feedback?>

## 14. Implementation-Plan Readiness

<Are contracts, ADRs, MVP slice, and handoff notes sufficient for
implementation planning?>

## 15. Blocking Issues

| # | Issue | Where (§) | Why It Blocks | Recommended Fix |
|---|---|---|---|---|
| 1 | <issue> | <§> | <impact> | <fix> |

> Blocking = prevents implementation planning or safe release. If none, write
> "No blocking issues."

## 16. Warnings

| # | Warning | Where (§) | Recommended Fix |
|---|---|---|---|
| 1 | <warning> | <§> | <fix> |

## 17. Polish Improvements

| # | Improvement | Where (§) |
|---|---|---|
| 1 | <polish item> | <§> |

## 18. Recommended Next Actions

<Ordered list tied to §15/§16. Which command should run next and why, e.g.:>

1. `architecture --mode update` — apply the selected stack (tech-stack says
   Architecture Update Required = Yes).
2. `architecture --mode reconcile` — absorb the ux-design architecture feedback.
3. `ux-design` / `implementation-plan` — once blocking issues are cleared.

## 19. Review Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Review is non-mutating | PASS / FAIL | architecture document not edited | — |
| Source architecture found | PASS / FAIL | <finding> | <action> |
| Optional artifacts handled correctly | PASS / WARNING / FAIL | <finding> | <action> |
| Scores are justified | PASS / WARNING / FAIL | <finding> | <action> |
| Issues are classified | PASS / WARNING / FAIL | blocking/warning/polish kept separate | <action> |
| Recommended next actions are clear | PASS / WARNING / FAIL | <finding> | <action> |

> Status legend: PASS / WARNING / FAIL. Never PASS a section with a known
> contradiction. Review must not silently rewrite the architecture.
