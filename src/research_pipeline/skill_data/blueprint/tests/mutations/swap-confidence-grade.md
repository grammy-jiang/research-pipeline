<!-- mutation: check=confidence_silently_upgraded level=FAIL detector=coherence needs-source-report
     Derived from a coherent base by swapping one confidence grade: the source
     report grades [2402.01234] MEDIUM, but this blueprint asserts it HIGH.
     Requires `--source-report mini_source_report.md`. Caught by
     `confidence_silently_upgraded`. -->

# Product Blueprint: Agent Memory (mutation — swapped confidence grade)

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [2. Workflow Model](#2-workflow-model)
- [3. MVP Scope](#3-mvp-scope)

---

## 1. Executive Product Thesis

Gated admission over durable agent memory. [2312.01234]

| Claim | Confidence | Citation |
|---|---|---|
| Selective forgetting is release-ready. | HIGH | [2402.01234] |

## 2. Workflow Model

### Workflow 1: Candidate Memory Admission

**Decision Gates:** redundant? (exact match) · contradicts existing?
On a contradiction the record is quarantined (fail-closed).

<!-- coherence: id=wf1.gate.contradicts stage=MVP-0 requires=wf1.servicer.quarantine -->
<!-- coherence: id=wf1.servicer.quarantine stage=MVP-0 -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed): contradictions quarantined.
