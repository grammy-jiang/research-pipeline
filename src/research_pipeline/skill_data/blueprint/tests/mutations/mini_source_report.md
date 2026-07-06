# Source Report (fixture for the mutation-derived negative set)

> Minimal source report used only by the `needs-source-report` mutations
> (e.g. `swap-confidence-grade.md`). It supplies a `## References` list and
> graded findings so `check_blueprint_coherence.py --source-report` can verify
> citation existence and detect silent confidence upgrades. Not a blueprint.

## Confidence-Graded Findings

- 🟢 HIGH — Evaluator-gated writes cut low-value memory. [2312.01234]
- 🟡 MEDIUM — Selective forgetting risks losing useful records. [2402.01234]

## References

- [2312.01234] Evaluator-gated memory writes.
- [2402.01234] Selective forgetting.
