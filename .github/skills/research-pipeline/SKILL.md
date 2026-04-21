---
name: research-pipeline
description: >
  Runs reproducible academic-paper research with research-pipeline: plan
  queries, search sources, screen papers, download/convert PDFs, extract
  evidence, summarize, validate, and write human-readable reports. Use when the
  user asks to find papers, search arXiv/Scholar/Semantic Scholar/OpenAlex/DBLP,
  conduct a literature review or paper survey, screen or summarize papers,
  expand citation graphs, evaluate paper quality, or build a research-backed
  system. Do not use for general web search or simple PDF reading.
metadata:
  author: grammy-jiang
  version: 1.9.0
  category: research
  tags: [arxiv, scholar, papers, research, literature-review, academic, citation-graph, quality-evaluation]
---

# Academic Paper Research

Use this skill to run the `research-pipeline` CLI/MCP workflow and produce
auditable literature-review artifacts. Keep `SKILL.md` light; load reference
files only when that detail is needed.

## Critical Rules

- Use the installed CLI directly: `research-pipeline`, not `uv run` or
  `python -m`, unless developing this repository itself.
- Always pass `--config` pointing at this skill's installed config:
  `~/.claude/skills/research-pipeline/config.toml` for Claude Code/GitHub
  Copilot, or `~/.codex/skills/research-pipeline/config.toml` for Codex.
  Use `CFG` as shorthand in commands below.
- Keep work evidence-based. Every substantive finding, recommendation, and
  theme in the final report must cite paper IDs or section-level evidence.
- Write the final human report to the current working directory:
  `./<topic-slug>-research-report.md`. Do not leave the final deliverable only
  under `runs/`.
- If an older `<topic-slug>-research-report.md` exists in the current directory,
  **resume on top of it, do not append**:
  1. Read the old report end-to-end. Extract prior paper IDs, themes,
     confidence levels, contradictions, and open gaps into working notes.
  2. Rename the old file to `<topic-slug>-research-report.<YYYY-MM-DD>.md`
     so it is preserved as a snapshot.
  3. Seed the new run: feed prior gaps and open questions into
     `query_plan.json` (as extra variants) and pass prior paper IDs as
     expansion seeds (`research-pipeline expand --paper-ids "...")` so new
     searches extend, rather than repeat, prior work.
  4. Regenerate the final report from scratch using the new run's artifacts
     plus the extracted prior context. The new report is a full replacement,
     not a diff or append.
- Human-facing reports must be easy to read and are REQUIRED to include:
  - A **table of contents** at the top (`## Contents`) with internal
    Markdown links to every section.
  - **Mermaid diagrams** for every process chart, architecture, or
    taxonomy (never ASCII art). Prefer vertical `flowchart TD`/`TB`.
  - **LaTeX** for every mathematical formula or equation, inline `$...$`
    or display `$$...$$` — do not render formulas as plain text.
  - Concise sections, short paragraphs, clear headings, tables for
    comparisons, and internal links from recommendations back to the
    findings, gaps, or evidence that support them.
- For system-building requests, evaluate implementation readiness and run
  iterative gap-filling when academic gaps remain.
- **Iterate to close gaps (up to 4 rounds).** After the first report is
  written and validated, inspect its gap sections. If any academic or
  engineering gaps remain, run another round: extract the gaps,
  translate each academic gap into a narrower pipeline iteration, fill
  engineering gaps from implementation knowledge, then regenerate the
  report from scratch (resume-on-top). Hard-cap at **4 rounds**; stop
  earlier if a regenerated report has no open gaps, a search returns
  no new relevant papers, or the user marks remaining gaps
  out-of-scope. See `references/iterative-synthesis.md` for the full
  loop.

## Load References

- `references/command-reference.md`: CLI options, source list, profiles, MCP
  tool mapping, and advanced utilities.
- `references/query-optimization.md`: use before editing `query_plan.json` or
  when search recall is weak.
- `references/sub-agents.md`: use when launching `paper-screener`,
  `paper-analyzer`, or `paper-synthesizer`.
- `references/output-templates.md`: use before writing the final report or
  validating report structure.
- `references/iterative-synthesis.md`: use for the mandatory 4-round
  gap-closure loop (applies to every run, not just system-building).
- `references/troubleshooting.md`: use for install/config/source/converter/MCP
  failures.

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
