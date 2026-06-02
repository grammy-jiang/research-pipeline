# Evaluation Strategy Template (§13)

The blueprint must define how to know whether the product works — without
assuming an implementation technology and without inventing performance
numbers.

## Evaluation table

| Evaluation | Purpose | Scenario | Expected Behaviour | Success Metric | MVP Required? | Traceability |
|---|---|---|---|---|---|---|
| Admission precision | Block low-value/poisoned writes | Feed mixed good/bad candidates | Bad candidates rejected or quarantined | Qualitative: no unsafe write admitted | Yes | [2312.01234] |
| Retrieval relevance | Surface the right records | Query with known-relevant set | Relevant records ranked above noise | Precision/recall, scenario-based | Yes | [Park et al., 2023] |

## Evaluation dimensions to consider

Task success · precision/recall · user utility · workflow completion ·
latency or operational cost (conceptual only — no numbers) · safety ·
security · robustness · abstention correctness · forgetting/deletion
correctness · human-review burden · regression risk.

## Rules

1. Every core capability needs at least one evaluation criterion.
2. Every HIGH-impact risk needs at least one evaluation or audit check.
3. If the source report contains benchmarks, methodology comparisons,
   benchmark tables, or paper-specific performance data, translate them
   into product evaluation scenarios.
4. If no benchmark exists, define a scenario-based evaluation harness.
5. Evaluation must not assume implementation technology.
6. For capabilities derived from an `ACADEMIC` gap, the evaluation
   scenario must explicitly measure whether the gap assumption holds.
