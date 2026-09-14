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
SKILL_DIR=~/.claude/skills/research-pipeline     # Claude Code
# SKILL_DIR=~/.copilot/skills/research-pipeline  # Copilot CLI
# SKILL_DIR=~/.agents/skills/research-pipeline   # Codex CLI
CFG=$SKILL_DIR/config.toml

research-pipeline-workflow "<topic>" --config "$CFG"
```

The runner reads `manifest.json`, initialises `workflow_state.json`,
and drives all tasks in dependency order. Each task status is written
to `workflow_state.json` before the next task begins. Completion requires successful execution receipts, schema and evidence checks,
reviewer acceptance where configured, and validation of the published content.

## Rules

1. **Use the runner and record execution results.** When a CLI/MCP stage is
   delegated, use research-pipeline-workflow --state STATE --execute-task ID.
   This captures the actual CLI exit code. For a direct MCP call, save its
   ToolResult JSON and submit --complete-task ID --result-file RESULT.json.
   A plan result must include artifacts.run_id; never guess the newest run.
2. **Resume with the existing state.** Accepted gates are revalidated. Old states
   without execution receipts stop for reconciliation; do not manufacture
   successful receipts. Search --resume reads saved candidates and coverage.
3. **Delegate using the printed contract.** A worker returns its artifacts and
   a result JSON containing success. Submit that result through the runner;
   never edit a task's status to accepted. Keep the user's requirements
   separate from your research hypotheses and delegated instructions.
4. **Honor failed gates and cooldowns.** Inspect the reason before using
   --retry-task ID. The runner invalidates dependents and limits retries.
   A reviewer rejection requires artifact correction and another independent
   review; it cannot be overridden by setting a status.
5. **Publish after validation.** The report task renders report/draft.md.
   Deep mode reviews that exact draft and its synthesis input. Validation
   records the draft hash; publish-report copies only that validated content
   to ./<topic-slug>-research-report.md.
6. **Check task fit before searching.** Choose date coverage from the question:
   a foundational survey needs older work; the default six-month window does
   not establish historical coverage. Inspect configured sources and optional
   dependencies, set plain-language source_queries where needed, and
   preserve the selected model/runtime. An analysis_model setting alone
   does not prove which model performed delegated work.
7. **Report evidence and gaps honestly.** Cite paper IDs from shortlist.json
   and analysis files. Read source_coverage.json: failed or cooling sources
   are coverage limitations, not evidence of no research. Evaluate academic
   and engineering gaps separately before another round. Record pipeline
   failures and the actual requests attempted.

## References

| File | Load when |
|------|-----------|
| `references/workflow-steps.md` | Per-agent SKILL_DIR paths, or understanding/diagnosing an orchestrated task |
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

## Required report format

Render Contents, Round History, a meaningful Mermaid diagram and LaTeX notation before review. Record only verified rounds; the built-in renderer describes the current synthesis snapshot and does not invent prior history. Invoke validation with --strict-format. Missing format elements must fail even if the weighted quality score passes. Publish only through the runner, which uses the active package interpreter and rechecks the report format against the validated content.
