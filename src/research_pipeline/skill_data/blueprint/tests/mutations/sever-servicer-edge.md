<!-- mutation: check=dangling_reference level=FAIL detector=coherence
     Derived from a coherent base by severing one servicing edge: the
     quarantine servicer anchor is deleted, so the MVP-0 contradiction gate
     now escalates to a servicer that is not declared anywhere (the def->use
     edge dangles). Caught by `dangling_reference`. -->

# Product Blueprint: Agent Memory (mutation — severed servicer edge)

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [2. Workflow Model](#2-workflow-model)
- [3. MVP Scope](#3-mvp-scope)

---

## 1. Executive Product Thesis

Gated admission over durable agent memory. [2312.01234]

## 2. Workflow Model

### Workflow 1: Candidate Memory Admission

**Decision Gates:** redundant? (exact match) · contradicts existing?
On a contradiction the record is quarantined (fail-closed).

<!-- coherence: id=wf1.gate.contradicts stage=MVP-0 requires=wf1.servicer.quarantine -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed): contradictions quarantined.
