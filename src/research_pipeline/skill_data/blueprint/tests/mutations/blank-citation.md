<!-- mutation: check=placeholder_citation level=WARNING detector=coherence
     Derived from a coherent base by blanking one citation: the thesis'
     arXiv id is replaced with an empty bracket pair. The coherence graph
     stays clean, so the only new finding is the placeholder-citation
     WARNING. -->

# Product Blueprint: Agent Memory (mutation — blank citation)

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [2. Workflow Model](#2-workflow-model)
- [3. MVP Scope](#3-mvp-scope)

---

## 1. Executive Product Thesis

Gated admission over durable agent memory. [] (citation blanked by mutation)

## 2. Workflow Model

### Workflow 1: Candidate Memory Admission

**Decision Gates:** redundant? (exact match) · contradicts existing?
On a contradiction the record is quarantined (fail-closed).

<!-- coherence: id=wf1.gate.contradicts stage=MVP-0 requires=wf1.servicer.quarantine -->
<!-- coherence: id=wf1.servicer.quarantine stage=MVP-0 -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed): contradictions quarantined.
