# Output Templates

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
