<!-- regression: check=open_dependency expect=PASS defect=precondition-currency
     The golden-fixture fix: the MVP-0 redundancy gate depends on exact-match
     dedup, an available MVP-0 servicer — not on the open question. Semantic
     dedup precision at repo scale stays a non-blocking scaling question that
     no MVP control requires, so scripts/check_blueprint_coherence.py must PASS.
     Paired with bad.md. -->

# Product Blueprint: Agent Memory (precondition-currency — FIXED)

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

**Decision Gates:** source trusted? · redundant? (exact match) · safe to store?
The redundancy gate uses exact-match dedup, which ships at MVP-0.

<!-- coherence: id=wf1.gate.redundant stage=MVP-0 requires=wf1.servicer.dedup -->
<!-- coherence: id=wf1.servicer.dedup stage=MVP-0 -->

## 3. MVP Scope

### 3.1 MVP-0 — Smallest Demonstrable Core

- Memory Admission workflow (fail-closed): exact-match dedup on the redundancy
  gate.

## 4. Open Questions and Validation Plan

| Question | Blocks MVP? |
|---|---|
| Semantic dedup precision at repo scale (beyond exact match) | No |

<!-- coherence: id=sec4.oq.dedup stage=open blocking=no -->

No MVP control depends on this open question, so the non-blocking scaling
concern does not gate MVP-0.
