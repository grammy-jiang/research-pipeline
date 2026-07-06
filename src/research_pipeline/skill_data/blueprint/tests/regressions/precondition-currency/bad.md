<!-- regression: check=open_dependency expect=FAIL defect=precondition-currency
     Reproduces the milder golden-fixture defect: an MVP-0 redundancy gate
     depends on cross-session dedup, but dedup is listed in Open Questions as
     non-blocking (`Blocks MVP? No`) with no phase qualifier — the gate's
     precondition is not current at MVP-0.
     scripts/check_blueprint_coherence.py must FAIL with `open_dependency`.
     Paired with fixed.md (MVP-0 exact-match dedup servicer). -->

# Product Blueprint: Agent Memory (precondition-currency — BAD)

## Contents

- [1. Executive Product Thesis](#1-executive-product-thesis)
- [2. Workflow Model](#2-workflow-model)
- [3. MVP Scope](#3-mvp-scope)
- [4. Open Questions and Validation Plan](#4-open-questions-and-validation-plan)

---

## 1. Executive Product Thesis

Gated admission over durable agent memory. [2312.01234]

## 2. Workflow Model

### Workflow 1: Candidate Memory Admission

**Decision Gates:** source trusted? · redundant? · safe to store?
The redundancy gate needs working cross-session dedup to update existing
records instead of duplicating them.

<!-- coherence: id=wf1.gate.redundant stage=MVP-0 requires=sec4.oq.dedup -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed), including the redundancy gate.

## 4. Open Questions and Validation Plan

| Question | Blocks MVP? |
|---|---|
| Dedup correctness across sessions | No |

<!-- coherence: id=sec4.oq.dedup stage=open blocking=no -->
