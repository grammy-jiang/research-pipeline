# Gap-Type Mapping

The `research-pipeline` skill classifies every open gap as `ACADEMIC`,
`ENGINEERING`, or `OUT_OF_SCOPE`. Each classification has a specific
product implication.

| Gap Classification | Product Action |
|---|---|
| `ENGINEERING` (HIGH severity) | Becomes a product requirement; candidate for MVP |
| `ENGINEERING` (MEDIUM severity) | Becomes a product requirement; candidate for Phase 2 |
| `ENGINEERING` (LOW severity) | Becomes a future extension |
| `ACADEMIC` (any severity) | Becomes a validation requirement or open question; must NOT become an MVP requirement unless the product's purpose is to answer the research question |
| `OUT_OF_SCOPE` | Note as a non-goal; do not include in the blueprint |

## Why the distinction matters

- An **engineering gap** is something the literature already knows how to
  solve in principle but no paper has productionized at the required
  scale/latency/robustness. It is fillable by engineering work, so it can
  become a product requirement.
- An **academic gap** is an open research question with no consensus. The
  product cannot assume it is solved. Treating it as a built feature would
  ship an unvalidated claim. Map it to a validation requirement, an open
  question, or a `DEFER / VALIDATE` decision.

## Consequences in later sections

- §6 decisions: `ENGINEERING` → ADOPT/ADAPT (sized by severity);
  `ACADEMIC` → DEFER / VALIDATE.
- §12 risk model: risks derived from `ACADEMIC` gaps must acknowledge that
  the research does not yet confirm the mitigation works.
- §13 evaluation: a capability derived from an `ACADEMIC` gap needs an
  evaluation scenario that explicitly measures whether the gap assumption
  holds.
- §14 MVP: `ACADEMIC`-gap items stay out of MVP unless the product's whole
  purpose is to validate that gap.
- §15 roadmap: an `ACADEMIC`-gap capability must not enter Phase 1 unless
  it is explicitly the product's validation purpose.
- §16 open questions: carry forward unresolved `ACADEMIC` gaps with a
  validation method.

## Worked examples

```text
ENGINEERING gap: "Deletion verification in production memory systems"
  → product requirement → Deletion & Forgetting Verification capability
  → MVP candidate (if HIGH severity and safety-relevant)

ACADEMIC gap: "Optimal memory consolidation frequency (no consensus)"
  → DEFER / VALIDATE → open question + evaluation scenario that measures
    consolidation quality before shipping any automatic scheduler

OUT_OF_SCOPE gap: "Hardware acceleration of embedding generation"
  → non-goal; excluded from the blueprint
```
