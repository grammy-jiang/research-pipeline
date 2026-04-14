# Output Templates

## Formatting Guidelines

All generated reports (final research report, synthesis, analysis) MUST use:

- **Mermaid** for diagrams and flowcharts (never ASCII art — it breaks easily)
- **LaTeX** for mathematical formulas and equations (inline `$...$` or display `$$...$$`)
- **Markdown tables** for structured data comparisons

Examples:

```markdown
<!-- Mermaid diagram -->
```mermaid
flowchart LR
    A[Search] --> B[Screen] --> C[Analyze]
```

<!-- LaTeX formula -->
The composite quality score is computed as:

$$Q = w_c \times C + w_v \times V + w_a \times A + w_r \times R$$

where $w_c = 0.35$, $w_v = 0.25$, $w_a = 0.25$, $w_r = 0.15$.
```

## Pipeline Status (after each stage)

```
## Pipeline Status — Run <RUN_ID>

| Stage | Status | Details |
|-------|--------|---------|
| Plan | ✅ Done | Topic: "...", N query variants |
| Search | ✅ Done | N candidates (arXiv: X, Scholar: Y) |
| Screen | ✅ Done | N → M shortlisted |
| Quality | ⬜ Optional | — |
| Expand | ⬜ Optional | — |
| Download | ✅ Done | N/N PDFs (size) |
| Convert | ✅ Done | N/N Markdown files |
| Extract | ⬜ Pending | — |
| Summarize | ⬜ Pending | — |

Run directory: runs/<RUN_ID>/
```

## Final Report Location

Write the final report to the **current working directory**:
```
./<topic-slug>-research-report.md
```
Example: `./local-memory-system-for-ai-agents-research-report.md`

NOT inside `runs/<run_id>/`.

## Final Summary (in chat)

```
## Research Summary — "<topic>"

**Run ID**: <RUN_ID>
**Sources**: arXiv, Google Scholar
**Timeline**: <start_time> → <end_time>
**Iterations**: <N> (if system-building mode)

### Pipeline Results
- **Searched**: <N> candidates from <sources>
- **Screened**: <N> → <M> shortlisted (top relevance: <score>)
- **Downloaded**: <N> PDFs (<size>)
- **Converted**: <N> Markdown files
- **Errors**: <list or "None">

### Key Findings
1. <finding with paper citation>
2. ...

### Top Papers
| # | Paper | Score | Key Contribution |
|---|-------|-------|------------------|
| 1 | <title> (arxiv_id) | <score> | <one-line summary> |

### Artifacts
- Final report: ./<topic-slug>-research-report.md
- Run data: runs/<RUN_ID>/
```
