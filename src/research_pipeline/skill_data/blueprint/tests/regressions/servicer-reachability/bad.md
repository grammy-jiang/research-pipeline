<!-- regression: check=phase_inversion expect=FAIL defect=servicer-reachability
     Reproduces the original golden-fixture defect: an MVP-0 admission
     contradiction gate escalates to a human-review servicer that is itself
     staged MVP-1, so the servicer is unavailable when the MVP-0 gate runs.
     scripts/check_blueprint_coherence.py must FAIL with `phase_inversion`.
     Paired with fixed.md (servicer pulled back to an MVP-0 quarantine). -->

# Product Blueprint: Agent Memory (servicer-reachability — BAD)

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
On a contradiction the gate escalates to human contradiction review before it
admits the record.

<!-- coherence: id=wf1.gate.contradicts stage=MVP-0 requires=sec3.review -->

## 3. Product Experience Direction

### 3.7 Human-in-the-Loop Experience

| Trigger | Expected Product Support | MVP Stage |
|---|---|---|
| Contradiction flagged | Human contradiction review | MVP-1 |

<!-- coherence: id=sec3.review stage=MVP-1 -->

## 4. MVP Scope

### 4.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed), including the contradiction gate.

### 4.2 MVP-1 — First Usable Version

- Human contradiction review.
