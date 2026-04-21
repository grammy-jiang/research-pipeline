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
  version: 1.10.0
  category: research
  tags: [arxiv, scholar, papers, research, literature-review, academic, citation-graph, quality-evaluation, gap-analysis]
---

# Academic Paper Research

Use this skill to run the `research-pipeline` CLI/MCP workflow and produce
auditable literature-review artifacts with evidence-cited, gap-closed
research reports. Keep `SKILL.md` light; load reference files only when
that detail is needed.

## When To Trigger

Trigger on requests like:

- "Research <topic>" / "Do a literature review on <topic>"
- "Find recent papers on <topic>" / "Survey arXiv for <topic>"
- "Build a research report on <topic>"
- "Screen / summarize / synthesize these papers"
- "Expand the citation graph for <paper>"
- "Resume the prior research report on <topic>"
- "Close the gaps in the research report"

Do **not** trigger for:

- General web search or news lookups (use web search).
- One-off PDF-to-Markdown conversion — direct the user to
  `research-pipeline convert-file <pdf>` instead, or the
  `pdf-to-markdown` skill.
- Requirements clarification, user stories, or architecture design —
  hand off to the `req-analysis` skill.
- Explaining a single paper the user already has — use the
  `paper-analyzer` sub-agent or `huggingface-papers` skill.

## Critical Rules

1. **Use the installed CLI directly.** `research-pipeline ...` — not
   `uv run` or `python -m`, unless developing this repository itself.
2. **Always pass `--config CFG`**, where `CFG` is the installed config:
   - `~/.claude/skills/research-pipeline/config.toml` (Claude Code / Copilot)
   - `~/.codex/skills/research-pipeline/config.toml` (Codex)
3. **Stay evidence-based.** Every substantive finding, recommendation,
   and theme in the final report must cite paper IDs or section-level
   evidence. `research-pipeline validate` enforces this.
4. **Final deliverable location.** Write the final human report to
   `./<topic-slug>-research-report.md` in the current working directory.
   Do not leave the final deliverable only under `runs/`.
5. **Resume on top of any prior report on the same topic — do not append.**
   If `./<topic-slug>-research-report.md` exists:
   1. Read it end-to-end. Extract prior paper IDs, themes, confidence
      levels, contradictions, and open gaps into working notes.
   2. Snapshot-rename the old file to
      `<topic-slug>-research-report.<YYYY-MM-DD>.md`.
   3. Seed the new run: add prior gaps/open questions as extra
      `query_variants`, and pass prior paper IDs via
      `research-pipeline expand --paper-ids "<ID1,ID2,...>"`. The
      global SQLite index dedups downloads and conversions across runs.
   4. Regenerate the final report from scratch. The new report fully
      replaces the old one — no diff, no append.
6. **Report formatting is REQUIRED, not optional:**
   - A `## Contents` table of contents at the top with internal
     Markdown links to every section.
   - A `## Round History` section right after `## Contents` with the
     per-round table (see `references/output-templates.md`).
   - **Mermaid** for every process chart, architecture, or taxonomy
     (never ASCII art). Prefer vertical `flowchart TD`/`TB`.
   - **LaTeX** for every formula, inline `$...$` or display `$$...$$`.
     Do not render formulas as plain text.
   - Concise sections, short paragraphs, clear headings, comparison
     tables, and internal links from recommendations back to findings,
     gaps, or evidence.
7. **Iterate to close gaps — hard cap 4 rounds.** After each report is
   written and validated, extract its gaps, classify them as
   `ACADEMIC` or `ENGINEERING`, and run another round (new pipeline
   iteration for academic gaps; filled from implementation knowledge
   for engineering gaps). Stop earlier when the report has no open
   gaps, a search returns no new relevant papers, or the user marks
   remaining gaps out-of-scope. See `references/iterative-synthesis.md`.
8. **For system-building requests**, evaluate implementation readiness
   (`IMPLEMENTATION_READY` / `HAS_GAPS` / `NOT_APPLICABLE`) and offer
   to hand off to the `req-analysis` skill only after the loop converges.

## Load References

Load each reference only when that level of detail is needed:

| File | When to load |
|------|--------------|
| `references/command-reference.md` | CLI options, sources, profiles, MCP tool map, advanced flags. |
| `references/query-optimization.md` | Before editing `query_plan.json` or when recall is weak. |
| `references/sub-agents.md` | Launching `paper-screener`, `paper-analyzer`, `paper-synthesizer`. |
| `references/output-templates.md` | Before writing or validating the final report. |
| `references/iterative-synthesis.md` | The mandatory 4-round gap-closure loop. |
| `references/troubleshooting.md` | Install/config/source/converter/MCP failures. |

## Default Workflow

Set:

```bash
# Claude Code / GitHub Copilot
CFG=~/.claude/skills/research-pipeline/config.toml

# Codex
# CFG=~/.codex/skills/research-pipeline/config.toml
```

1. **Check for a prior report on the same topic (resume, don't restart)**
   - Look for `./<topic-slug>-research-report.md` in the working directory.
   - If **present**:
     1. Read it fully. Extract prior paper IDs, main themes, confidence
        levels, contradictions, and open gaps into working notes.
     2. Rename the existing file to
        `<topic-slug>-research-report.<YYYY-MM-DD>.md` as a snapshot.
     3. Seed the new pipeline run with that context:
        - Add prior unanswered gaps and open questions as extra
          `query_variants` in the new `query_plan.json`.
        - Pass prior paper IDs to
          `research-pipeline expand --paper-ids "<ID1,ID2,...>"` so
          search extends (not duplicates) the earlier shortlist.
        - The global SQLite paper index deduplicates across runs, so
          prior papers are not re-downloaded or re-converted unless
          explicitly re-requested.
     4. When writing the final report, **regenerate from scratch** using
        the new run's artifacts plus the extracted prior context. The
        new report fully replaces the old one — do not append, merge,
        or diff.
   - If **absent**: proceed with a fresh run.

2. **Plan and improve queries**
   ```bash
   research-pipeline plan "topic" --config CFG
   research-pipeline verify --run-id <RUN_ID> --stage plan --config CFG
   ```
   Review `runs/<run_id>/plan/query_plan.json`. Keep `must_terms` to 2-3
   terms, add synonym-rich `query_variants`, and consult
   `references/query-optimization.md` when recall matters.

3. **Search and screen**
   ```bash
   research-pipeline search --run-id <RUN_ID> --source all --config CFG
   research-pipeline screen --run-id <RUN_ID> --diversity --config CFG
   ```
   `--source all` searches arXiv, Scholar, Semantic Scholar, OpenAlex, DBLP,
   and HuggingFace daily papers. Use `paper-screener` when BM25 results are
   noisy or the topic has broad terms.

4. **Optional expansion and organization**
   ```bash
   research-pipeline quality --run-id <RUN_ID> --config CFG
   research-pipeline expand --run-id <RUN_ID> --paper-ids "ID1,ID2" --direction both --config CFG
   research-pipeline cluster --run-id <RUN_ID>
   research-pipeline enrich --run-id <RUN_ID> --config CFG
   ```
   Use `expand` for citation graph recall, `quality` for citation/venue/author
   signals, `cluster` for topical organization, and `enrich` when metadata is
   missing.

5. **Download, convert, extract**
   ```bash
   research-pipeline download --run-id <RUN_ID> --config CFG
   research-pipeline convert-rough --run-id <RUN_ID> --config CFG
   research-pipeline convert-fine --run-id <RUN_ID> --paper-ids "ID1,ID2" --config CFG
   research-pipeline extract --run-id <RUN_ID> --config CFG
   ```
   For small runs, `research-pipeline convert --run-id <RUN_ID> --config CFG`
   is fine. For larger runs, rough-convert all PDFs and fine-convert selected
   papers.

6. **Summarize, analyze, and synthesize**
   ```bash
   research-pipeline summarize --run-id <RUN_ID> --config CFG
   research-pipeline analyze-claims --run-id <RUN_ID>
   research-pipeline score-claims --run-id <RUN_ID>
   ```
   `summarize` is a two-step structured workflow. Step 1 writes
   `summarize/extractions/*.extraction.json` and `.md` files with typed
   statements, evidence snippets, confidence labels, uncertainty notes, and
   extraction quality scores. Step 2 consumes those records to write
   `summarize/synthesis_report.json`, `synthesis_report.md`,
   `synthesis_traceability.json`, and legacy `synthesis.json`.
   For deep work, launch sub-agents after conversion: optional
   `paper-screener`, one `paper-analyzer` per important paper, then
   `paper-synthesizer`. Use the strongest available reasoning model for
   sub-agents.

7. **Write, validate, and export the final report**
   ```bash
   research-pipeline report --run-id <RUN_ID> --template structured_synthesis --config CFG
   research-pipeline validate --report ./<topic-slug>-research-report.md --config CFG
   research-pipeline export-html --markdown ./<topic-slug>-research-report.md -o ./<topic-slug>-research-report.html --config CFG
   research-pipeline export-bibtex --run-id <RUN_ID> --stage screen -o refs.bib
   ```
   Before writing, load `references/output-templates.md`. The report MUST
   contain:
   - A `## Contents` section at the top with internal Markdown links to
     every section (table of contents).
   - **Mermaid diagrams** (vertical `flowchart TD`/`TB`) for process
     flows, architectures, and taxonomies — never ASCII art.
   - **LaTeX** for every formula (`$...$` inline, `$$...$$` display).
   - Confidence levels, evidence citations, gap classifications, tables
     for comparisons, and internal links from recommendations back to
     findings or gaps.

8. **Diagnose where quality comes from (optional but recommended)**
   Two evaluation commands operationalize remaining-gap signals from the
   Deep Research Report:
   ```bash
   # A3-5 Unified Horizon Metric: single scalar combining quality, difficulty,
   # horizon length, entropy stability, and Pass[k] reliability.
   research-pipeline horizon --score 0.8 --achieved 40 --target 50

   # Theme 16 Recall/Reasoning/Presentation diagnostic: localize the
   # bottleneck axis of a synthesis report.
   research-pipeline rrp --report report.md --shortlist shortlist.json
   ```
   If `rrp` reports `recall` as the bottleneck, rerun screening or expand
   the candidate set before re-writing prose; presentation-bottleneck runs
   usually only need template/format polish.

9. **Iterate to close gaps (up to 4 rounds)**
   Open `./<topic-slug>-research-report.md` and inspect its gap
   sections (`Research Gaps`, `Unresolved Questions`, `Assumption Map`,
   `Risk Register`). If every gap is empty or marked as accepted
   limitation, the loop converged in 1 round — stop and report.
   Otherwise run another round:
   - Classify each open gap as `ACADEMIC` or `ENGINEERING`.
   - Fill `ENGINEERING` gaps from implementation knowledge; record
     the resolutions inline inside the regenerated report.
   - For each `ACADEMIC` gap (or small cluster of related gaps),
     derive a narrower search question and start a new pipeline run
     (steps 2–7 above) seeded with prior paper IDs via
     `research-pipeline expand --paper-ids "<ID1,ID2,...>"`. The
     global SQLite index dedups downloads and conversions across
     rounds automatically.
   - Snapshot-rename the existing report to
     `<topic-slug>-research-report.<YYYY-MM-DD>.md`, regenerate from
     scratch using the combined corpus, and update the report's
     `## Round History` table at the top.
   Stop iterating when any of these are true: regenerated report has
   no open gaps; **4 rounds** completed; new search returned zero
   new relevant papers; remaining gaps are user-marked out-of-scope.
   Never exceed 4 rounds. See `references/iterative-synthesis.md` for
   the full loop.

## Profiles

Use profiles when running end-to-end:

| Profile | Use When |
|---|---|
| `quick` | Fast abstract-only overview |
| `standard` | Normal evidence-backed review |
| `deep` | Comprehensive review with quality, expansion, claim analysis, and TER gap filling |
| `auto` | Let the CLI choose based on query complexity |

```bash
research-pipeline run --profile standard "topic" --source all --config CFG
research-pipeline run --profile deep "topic" --source all --ter-iterations 3 --config CFG
```

## System-Building Mode

If the user wants to build, design, implement, or architect a system, apply
the iterative gap-closure loop above with these additional constraints:

1. Require a readiness verdict:
   `IMPLEMENTATION_READY`, `HAS_GAPS`, or `NOT_APPLICABLE`.
2. Classify gaps as `ACADEMIC` or `ENGINEERING` (the 4-round loop
   handles both).
3. Do not hand off to `req-analysis` until the loop converges
   (`IMPLEMENTATION_READY`, no open gaps, or 4 rounds used).

Load `references/iterative-synthesis.md` for the full loop (it
applies to every run, not only system-building).

## MCP Usage

The MCP server exposes the same workflow. Prefer `tool_research_workflow` for
server-driven orchestration, or use stage tools such as `tool_plan_topic`,
`tool_search`, `tool_screen_candidates`, `tool_report`, and
`tool_validate_report`. See `references/command-reference.md` for the current
tool map.

## Final Response To User

After completing a run, report:

- final report path;
- run ID(s) and profile (list every round's run ID if multiple);
- **number of gap-closure rounds executed (of the 4-round cap) and
  why iteration stopped** (converged / cap reached / no new papers /
  out-of-scope);
- source counts and shortlist/download/convert totals;
- top findings with confidence and citations;
- validation result and any remaining gaps;
- important artifacts such as HTML, BibTeX, or comparison outputs.

## Examples

Concrete user request → skill behavior. Use these to calibrate
triggering and default actions.

### Example 1 — Fresh literature review

User: *"Research recent work on transformer architectures for time series."*

Actions:

1. No prior `./transformer-time-series-research-report.md` exists.
2. Plan → search (`--source all`) → screen → download → convert →
   extract → summarize → report, all with `--config CFG`.
3. Validate the report, run RRP diagnostic if quality is borderline.
4. Inspect gaps; if any remain, iterate (rounds 2-4).
5. Deliver `./transformer-time-series-research-report.md` (with
   `## Contents`, `## Round History`, Mermaid, LaTeX).

### Example 2 — Resume a prior report

User: *"Update my ai-agent-memory research report with the latest papers."*

Actions:

1. Read `./ai-agent-memory-research-report.md` end-to-end; extract
   prior paper IDs, themes, and open gaps.
2. Snapshot-rename to `ai-agent-memory-research-report.2026-04-22.md`.
3. Run a new pipeline round seeded with prior paper IDs via
   `research-pipeline expand --paper-ids "<IDs>"` and prior gaps as
   extra `query_variants`.
4. Regenerate the report from scratch (not appended); continue
   iterating up to 4 rounds until gaps close.

### Example 3 — System-building goal

User: *"I want to build a local memory system for AI agents. Research the state of the art."*

Actions:

1. Run Example 1's workflow.
2. After the first report, require an implementation-readiness verdict
   (`IMPLEMENTATION_READY` / `HAS_GAPS` / `NOT_APPLICABLE`).
3. Iterate the 4-round loop until the verdict flips to
   `IMPLEMENTATION_READY` or the cap is hit.
4. Once converged, offer to hand off to the `req-analysis` skill for
   requirements → architecture work.

### Example 4 — Out of scope, redirect

User: *"Convert this one PDF to Markdown."*

Actions:

1. Do **not** invoke the full pipeline.
2. Direct the user to `research-pipeline convert-file <pdf> -o <out>`
   or the `pdf-to-markdown` skill.
