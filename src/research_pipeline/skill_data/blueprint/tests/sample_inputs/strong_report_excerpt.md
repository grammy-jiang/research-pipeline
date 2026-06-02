# Research Report: Memory Systems for AI Coding Agents

> Illustrative excerpt used to exercise the `blueprint` skill. Trimmed to
> the sections the skill relies on. Citations are fictional.

## Executive Summary

Persistent memory measurably improves multi-session AI coding agents, but
naive memory accumulates low-value and unsafe entries. The strongest
results come from systems that gate writes, scope records, and retrieve
with hybrid keyword+semantic search. Deletion correctness and cross-session
deduplication remain under-productionized.

## Research Question

How should a local-first memory system for AI coding agents admit, store,
retrieve, and forget knowledge across sessions while resisting poisoning
and leakage?

## Methodology

12 papers screened from arXiv, Semantic Scholar, and OpenAlex (2022–2025);
8 retained after BM25 + SPECTER2 screening. Two reviewer passes.

## Research Landscape

- **Theme 1 — Memory taxonomy:** episodic vs semantic vs working memory.
- **Theme 2 — Admission control:** evaluator-gated writes reduce noise.
- **Theme 3 — Retrieval:** hybrid BM25 + dense retrieval beats either alone.
- **Theme 4 — Forgetting:** selective forgetting and tombstoning.

## Confidence-Graded Findings

- 🟢 HIGH — Evaluator-gated writes cut low-value memory substantially
  across 3 systems [2312.01234], [2401.05678], [Park et al., 2023].
- 🟢 HIGH — Hybrid BM25 + dense retrieval improves recall over either
  alone [Park et al., 2023], [2310.05338], [2402.07788].
- 🟢 HIGH — Scoped memory (local/project/team) limits leakage [2401.05678].
- 🟡 MEDIUM — Pyramid/hierarchical retrieval helps at large scale, with
  caveats [2401.05679].
- 🟡 MEDIUM — Selective forgetting improves signal but risks losing useful
  records [2402.01234].
- 🔴 LOW — Fully automatic consolidation scheduling is promising but
  preliminary [2403.09999].

## Research Gaps

- **ENGINEERING (HIGH):** Deletion verification in production memory
  systems — no paper demonstrates auditable hard-delete at scale.
- **ENGINEERING (MEDIUM):** Cross-session deduplication at repository
  scale.
- **ACADEMIC:** Optimal memory consolidation frequency — no consensus.
- **OUT_OF_SCOPE:** Hardware acceleration of embedding generation.

## Points of Contradiction

- Aggressive forgetting (recall vs. storage cost) [2402.01234] vs.
  retain-everything indexing [2401.05679].

## Practical Recommendations

- Gate every write behind an admission evaluator.
- Default to scoped records; require explicit promotion to widen scope.
- Provide an audit trail for every admission and deletion decision.

## Readiness Assessment

`HAS_GAPS` — core read/write/admission paths are implementation-ready;
deletion verification and consolidation need validation.

## Round History

| Round | New papers | Gaps closed | Gaps remaining |
|---|---|---|---|
| 1 | 8 | 2 | 4 |
| 2 | 3 | 2 | 2 |

## References

- [2312.01234] Evaluator-gated memory writes.
- [2401.05678] Scoped agent memory and promotion.
- [2310.05338] Memory taxonomy for agents.
- [Park et al., 2023] Hybrid retrieval for agent memory.
- [2402.07788] Dense + sparse retrieval fusion.
- [2401.05679] Hierarchical/pyramid retrieval.
- [2402.01234] Selective forgetting.
- [2403.09999] Automatic memory consolidation.
