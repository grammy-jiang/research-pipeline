---
name: research-pipeline
description: >
  End-to-end reproducible academic-paper research using the
  research-pipeline CLI and MCP server. Plans queries, searches
  arXiv/Google Scholar/Semantic Scholar/OpenAlex/DBLP/HuggingFace,
  screens with BM25, downloads and converts PDFs to Markdown,
  extracts evidence, summarizes, validates, and writes a
  human-readable, evidence-cited research report. Iterates up to
  4 rounds to close academic and engineering gaps, and resumes on
  top of any prior same-topic report. Use when the user asks to
  "research", "find papers", "survey the literature", "do a
  literature review", "analyze arXiv papers", "build a research
  report", "screen papers", "summarize papers", "expand citations",
  "fill research gaps", or "resume a prior research report". Do
  NOT use for general web search, simple PDF-to-Markdown conversion
  (use `research-pipeline convert-file` directly), or requirements
  analysis (use the req-analysis skill).
license: MIT
compatibility: >
  Requires the research-pipeline CLI (pip install research-pipeline)
  and network access to arXiv/Scholar/Semantic Scholar/OpenAlex/DBLP.
  Works in Claude Code, Claude.ai, GitHub Copilot, and Codex.
metadata:
  author: grammy-jiang
  version: 2.0.0
  category: research
  tags: [arxiv, scholar, papers, research, literature-review, academic, citation-graph, quality-evaluation, gap-analysis]
---

# Academic Paper Research

## When To Trigger

- "Research `<topic>`" / "Do a literature review on `<topic>`"
- "Find recent papers on `<topic>`" / "Survey arXiv for `<topic>`"
- "Build a research report on `<topic>`" / "Screen / summarize / synthesize papers"
- "Expand the citation graph" / "Resume the prior research report"
- "Close the gaps in the research report"

Do **not** trigger for general web search, single PDF conversion
(`research-pipeline convert-file`), requirements/architecture design
(use `req-analysis`), or single-paper explanation (use `paper-analyzer`).

## Launch

**Always launch through the manifest-governed runner. Never bypass it.**

```bash
SKILL_DIR=~/.claude/skills/research-pipeline    # Claude Code / Copilot
# SKILL_DIR=~/.codex/skills/research-pipeline   # Codex
CFG=$SKILL_DIR/config.toml

python3 $SKILL_DIR/runners/runner.py "<topic>" --config "$CFG"
```

The runner reads `manifest.json`, initialises `workflow_state.json`,
and drives all tasks in dependency order. Each task status is written
to `workflow_state.json` before the next task begins. Completion is
proved by artifact existence + schema validation — not by agent claim.

## Rules

1. **Do not bypass the runner.** Never call pipeline CLI commands
   directly without the orchestrator updating `workflow_state.json`.
2. **Resume = re-run the runner.** Pass `--state <existing>.json` to
   continue an interrupted workflow. Idempotent: accepted tasks are skipped.
3. **Sub-agent delegation.** When the runner prints `DELEGATE TO SUB-AGENT`,
   execute the named sub-agent with the printed contract, then update
   `workflow_state.json tasks.<id>.status = "accepted"` and re-run.
4. **Reviewer gates.** If a reviewer sub-agent returns `status: "rejected"`,
   fix the artifact, reset the task to `pending`, and re-run. Do not
   override a `rejected` verdict.
5. **Final report.** Write `./<topic-slug>-research-report.md` only after
   `validate-report` is `accepted`. Never write it before that gate.
6. **Evidence-based.** Every finding in the report must cite at least one
   paper ID traceable to `screened.jsonl` or `analysis/`.

## References

| File | Load when |
|------|-----------|
| `references/command-reference.md` | CLI options, profiles, MCP tool map, advanced flags |
| `references/query-optimization.md` | Editing `query_plan.json` or weak recall |
| `references/sub-agents.md` | Launching screener, analyzer, synthesizer |
| `references/output-templates.md` | Writing or validating the final report |
| `references/iterative-synthesis.md` | Gap-closure rounds and stopping conditions |
| `references/troubleshooting.md` | Install/config/source/converter/MCP failures |

## Final Response To User

When `workflow_state.json` shows `status: complete`:

1. Show the final report path and the round-history table.
2. List any remaining open gaps (ACADEMIC / ENGINEERING) not closed this run.
3. Offer to run another round, expand citations, or hand off to `req-analysis`.
