# Borderline Cases: Implementation Neutrality

The blueprint must stay implementation-neutral. This file is the
reference for deciding whether a statement is allowed.

## Clearly allowed (conceptual behaviour / responsibility)

```text
The product needs a durable record store.
The product needs an admission-control workflow.
The product needs an agent integration surface.
The product needs a retrieval workflow.
The product needs an evaluation harness.
The product needs a governance layer.
```

## Clearly forbidden (named technology / vendor / deployment / code)

```text
Use Python.            Use FastAPI.          Use PostgreSQL.
Use Chroma.            Use React.            Use AWS.
Create table memory_records.   Implement class MemoryAdmissionController.
Install package X.     Deploy with Docker Compose.
```

## Borderline table

| Statement | Verdict | Reason |
|---|---|---|
| Append-only event ledger | ✅ Allowed | Conceptual behaviour, not a technology |
| Distributed messaging layer | ✅ Allowed | Conceptual responsibility, not a vendor |
| The product needs Kafka | ❌ Forbidden | Named technology |
| A document store | ✅ Allowed | Conceptual storage type |
| The product needs MongoDB | ❌ Forbidden | Named vendor |
| Efficient vector similarity search | ✅ Allowed | Conceptual capability |
| The product needs FAISS | ❌ Forbidden | Named library |
| Deployed as microservices | ⚠️ Borderline | Leans toward deployment architecture; defer to technical design |
| Process isolation between tenants | ✅ Allowed | Conceptual security boundary, not an infra choice |

## Decision rule

> If removing the specific term and replacing it with its **purpose**
> still conveys the constraint clearly, the conceptual version is
> preferred. If the statement cannot be expressed without naming a
> specific product, vendor, or deployment model, it belongs in the
> technical-architecture skill.

## Warning tier (runtime-leaning, not forbidden)

Not every implementation-leaning phrase is a hard failure. Use four levels:

| Level | Meaning | Example | Action |
|---|---|---|---|
| Allowed | Conceptual product responsibility | quality gate, domain profile, audit record, durable store | Keep |
| Warning | Runtime / architecture / deployment leaning | "service-deployable", "deployed as microservices", a specific context-window assumption, "reachable model" | Rephrase to its purpose, or defer to §17 technical-design handoff; record as a `WARNING` in the self-check |
| Research-derived exception | A specific mechanism named in the research | a cited decoding/caching technique used as an evaluation anchor | Keep only if cited and tied to evaluation |
| Forbidden | Concrete technology, vendor, or deployment product | Python, FastAPI, PostgreSQL, FAISS, AWS, Docker | `FAIL` — remove |

A `WARNING`-level phrase does not fail the blueprint, but it must be made
visible (rephrased, or surfaced in the Appendix A self-check) rather than
silently accepted. When in doubt between Warning and Forbidden, prefer the
more conservative classification.

## Conceptual component names

Neutral logical names such as `Memory Admission Controller` or `Retrieval
Orchestrator` are allowed as responsibility boundaries. Do **not** turn
them into concrete source-code modules, classes, services, or deployable
units unless a downstream technical-design task explicitly asks for it.
See `templates/logical_architecture_template.md` for the allowed/forbidden
name lists.
