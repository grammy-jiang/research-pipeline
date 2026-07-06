<!--
Minimal COHERENT blueprint fixture for scripts/check_blueprint_coherence.py.
Every MVP-0 node's required servicer is available no later than the node, and
no MVP node depends on a non-blocking open question. The guard must PASS.
Companion: incoherent_blueprint.md (same shape, phase-inverted staging).
-->

# Product Blueprint: Coherent Memory Service

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
| Contradiction flagged | Human contradiction review | MVP-0 |

<!-- coherence: id=sec3.r1 stage=MVP-0 -->

## 4. MVP Scope

### 4.3 Safety Baseline

- Contradiction review is available before Admission can flag, so the MVP-0
  escalation path has a servicer. [Park et al., 2023]

## 5. Open Questions and Validation Plan

| Question | Blocks MVP? |
|---|---|
| Optimal dedup similarity threshold | No |

<!-- coherence: id=sec5.oq1 stage=open blocking=no -->
