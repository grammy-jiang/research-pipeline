<!--
Minimal INCOHERENT blueprint fixture for scripts/check_blueprint_coherence.py.
It reproduces both known defect classes from issue #81:
  1. phase_inversion  — an MVP-0 workflow gate whose servicer is staged MVP-1.
  2. open_dependency  — an MVP-0 "non-negotiable" control gated on a
                        non-blocking open question, with no phase qualifier.
The guard must FAIL. Companion: coherent_blueprint.md (corrected staging).
-->

# Product Blueprint: Incoherent Memory Service

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [2. Workflow Model](#2-workflow-model)
- [3. Product Experience Direction](#3-product-experience-direction)
- [4. MVP Scope](#4-mvp-scope)
- [5. Open Questions and Validation Plan](#5-open-questions-and-validation-plan)

---

## 1. Executive Product Thesis

A minimal fixture exercising the deterministic cross-phase coherence guard.
[Park et al., 2023]

## 2. Workflow Model

### Workflow 1: Admission

**Decision Gates:**
1. If the incoming record contradicts stored state, escalate to contradiction
   review before admitting it.

<!-- coherence: id=wf1.gate.contradicts stage=MVP-0 requires=sec3.r1 -->

## 3. Product Experience Direction

### 3.7 Human-in-the-Loop Experience

| Trigger | Expected Product Support | MVP Stage |
|---|---|---|
| Contradiction flagged | Human contradiction review | MVP-1 |

<!-- coherence: id=sec3.r1 stage=MVP-1 -->

## 4. MVP Scope

### 4.3 Safety Baseline

- Contradiction review is a non-negotiable MVP-0 control, but its threshold is
  still an open question. [Park et al., 2023]

<!-- coherence: id=sec4.c1 stage=MVP-0 requires=sec5.oq1 -->

## 5. Open Questions and Validation Plan

| Question | Blocks MVP? |
|---|---|
| Optimal dedup similarity threshold | No |

<!-- coherence: id=sec5.oq1 stage=open blocking=no -->
