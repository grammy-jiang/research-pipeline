<!-- regression: check=phase_inversion expect=PASS defect=servicer-reachability
     The golden-fixture fix: at MVP-0 a contradiction is quarantined (an MVP-0
     servicer, fail-closed); human contradiction review is an MVP-1 enhancement
     that consumes the MVP-0 quarantine. No servicer is staged after the gate
     that requires it, so scripts/check_blueprint_coherence.py must PASS.
     Paired with bad.md. -->

# Product Blueprint: Agent Memory (servicer-reachability — FIXED)

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [2. Workflow Model](#2-workflow-model)
- [3. Product Experience Direction](#3-product-experience-direction)
- [4. MVP Scope](#4-mvp-scope)

---

## 1. Executive Product Thesis

Gated admission over durable agent memory. [2312.01234]

## 2. Workflow Model

### Workflow 1: Candidate Memory Admission

**Decision Gates:** source trusted? · redundant? · contradicts existing?
On a contradiction the gate quarantines the record (fail-closed); human
contradiction review is deferred to MVP-1.

<!-- coherence: id=wf1.gate.contradicts stage=MVP-0 requires=wf1.servicer.quarantine -->
<!-- coherence: id=wf1.servicer.quarantine stage=MVP-0 -->

## 3. Product Experience Direction

### 3.7 Human-in-the-Loop Experience

| Trigger | Expected Product Support | MVP Stage |
|---|---|---|
| Contradiction flagged | Human contradiction review over quarantined items | MVP-1 |

<!-- coherence: id=sec3.review stage=MVP-1 requires=wf1.servicer.quarantine -->

## 4. MVP Scope

### 4.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed): contradictions quarantined at MVP-0.

### 4.2 MVP-1 — First Usable Version

- Human contradiction review over the MVP-0 quarantined contradictions.
