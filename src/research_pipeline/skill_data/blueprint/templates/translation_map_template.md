# Translation Map & Decision Templates (§5 and §6)

## §5 Research-to-Product Translation Map

One row per extracted research item that is at least `useful` relevance.
Order by relevance (critical → useful), then by confidence.

| Research Item | Type | Confidence | Product Primitive | Product Relevance | Citation |
|---|---|---|---|---|---|
| Evaluator-gated writes | mechanism | HIGH 🟢 | Memory Admission Workflow | critical | [2312.01234] |
| Hybrid retrieval | mechanism | MEDIUM 🟡 | Hybrid Retrieval Capability | useful | [Park et al., 2023] |

Item types: taxonomy, mechanism, algorithm, workflow_pattern, benchmark,
security_method, data_structure, empirical_result, assumption,
contradiction, academic_gap, engineering_gap, risk,
operational_implication, architecture_hint.

## §6 Adopt / Adapt / Merge / Defer / Reject Decisions

One row per major idea. Every row needs a citation **or** an explicit
design-decision rationale.

| Source Idea | Citation | Decision | Product Translation | Rationale | MVP? |
|---|---|---|---|---|---|
| Idea A | [2312.01234] | ADOPT | Capability X | HIGH confidence, central to product | Yes |
| Idea B | [Park, 2023] | MERGE | Capability Y | Overlaps Ideas C and D | Yes |
| Idea E | [2401.05678] | DEFER | Future Extension Z | Valuable but not MVP-critical | No |
| Idea F | [LOW confidence] | REJECT | None | Evidence weak or out of scope | No |
| Idea G | [ACADEMIC gap] | DEFER / VALIDATE | Open question / validation | Research does not yet confirm | No |

Decision reminders:

- HIGH + product-critical → usually ADOPT / MERGE.
- MEDIUM → usually ADAPT / DEFER.
- LOW → usually DEFER unless cheap and safe.
- `ACADEMIC` gaps → DEFER / VALIDATE (never MVP unless the product's
  purpose is to validate that gap).
- Security-critical gaps → risk control or release gate.
