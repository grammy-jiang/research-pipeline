<!-- mutation: term=postgresql level=FAIL detector=neutrality
     Derived from a coherent base by inserting one forbidden tech-stack term
     (a concrete database product). This is an implementation-neutrality
     violation (forbidden_content_checklist.md / Gate 3), NOT a cross-phase
     coherence defect — the deterministic guard does not and should not catch
     it; the neutrality gate does. -->

# Product Blueprint: Agent Memory (mutation — forbidden tech-stack term)

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
<!-- coherence: id=wf1.servicer.quarantine stage=MVP-0 -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed) persisted in PostgreSQL. <!-- forbidden tech-stack selection inserted by mutation -->
