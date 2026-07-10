<!-- mutation: check=vendor_leak level=FAIL detector=coherence
     Derived from a coherent base by inserting one named vendor CLI plus its
     wire-level config flags into a single MVP-0 bullet (see the last line).
     This is an implementation-neutrality leak the LLM Gate 3 demonstrably
     missed and then restated PASS for; the deterministic guard now re-derives
     the neutrality verdict from a fresh body scan and FAILs (`vendor_leak`),
     so the Appendix A row is a computed result, not a stale generation-time
     claim. Uncited vendor tokens leak — a line that cites a paper is allowed. -->

# Product Blueprint: Agent Memory (mutation — vendor CLI + config-flag leak)

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

- Memory Admission workflow gated by configuring Claude Code `permissions.deny`
  and `--sandbox`. <!-- named vendor CLI + wire-level config leak inserted by mutation -->
