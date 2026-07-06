# Mutation-derived negative fixtures

Each `*.md` here (except this file and `mini_source_report.md`) is a **single**
mutation of a minimal *coherent* blueprint base — one defect per file — so a
reviewer or CI check can prove the blueprint quality gates actually reject a
violating document. Companion to the coherent golden fixture
(`../sample_outputs/product_blueprint_example.md`), the `../coherence_fixtures/`
pair, and the `../regressions/` pairs.

Every mutation file begins with a machine-readable label naming the check that
must catch it:

```text
<!-- mutation: check=<check> level=<FAIL|WARNING> detector=<coherence|neutrality>
              [needs-source-report] [term=<forbidden-term>] ... -->
```

`tests/unit/test_skill_blueprint.py` discovers these files (any file whose first
line is a `<!-- mutation: -->` label) and asserts each mutation triggers exactly
its labelled check.

| File | Mutation | Detector | Expected check |
|------|----------|----------|----------------|
| `invert-mvp-tag.md` | flip a servicer MVP-0 → MVP-1 | coherence guard | `phase_inversion` (FAIL) |
| `sever-servicer-edge.md` | delete a required servicer anchor | coherence guard | `dangling_reference` (FAIL) |
| `blank-citation.md` | blank one citation to `[]` | coherence guard | `placeholder_citation` (WARNING) |
| `swap-confidence-grade.md` | upgrade one citation's confidence vs. source | coherence guard `--source-report` | `confidence_silently_upgraded` (FAIL) |
| `forbidden-term.md` | insert a concrete tech-stack product | neutrality gate (Gate 3) | forbidden term present (guard does *not* catch) |

`mini_source_report.md` supplies the `## References` + graded findings that the
`needs-source-report` mutations are checked against.
