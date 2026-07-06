<!-- mutation: check=phase_inversion level=FAIL detector=coherence
     Derived from a coherent base by inverting one MVP tag: the quarantine
     servicer is flipped from MVP-0 to MVP-1, so the MVP-0 contradiction gate
     now escalates to a later-staged servicer. Caught by `phase_inversion`. -->

# Product Blueprint: Agent Memory (mutation — inverted MVP tag)

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
<!-- coherence: id=wf1.servicer.quarantine stage=MVP-1 -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed): contradictions quarantined.
