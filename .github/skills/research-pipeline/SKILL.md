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
  version: 1.7.0
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
- If an older `<topic-slug>-research-report.md` exists, read it for prior
  context, rename it with a date suffix, and generate a fresh report. Do not
  append-merge into old reports.
- Human-facing reports must be easy to read: concise sections, short
  paragraphs, clear headings, internal Markdown links, Mermaid diagrams
  (prefer vertical `flowchart TD`/`TB`), LaTeX for formulas, and tables for
  comparisons.
- For system-building requests, evaluate implementation readiness and run
  iterative gap-filling when academic gaps remain.

## Load References

- `references/command-reference.md`: CLI options, source list, profiles, MCP
  tool mapping, and advanced utilities.
- `references/query-optimization.md`: use before editing `query_plan.json` or
  when search recall is weak.
- `references/sub-agents.md`: use when launching `paper-screener`,
  `paper-analyzer`, or `paper-synthesizer`.
- `references/output-templates.md`: use before writing the final report or
  validating report structure.
- `references/iterative-synthesis.md`: use for system-building goals or
  `HAS_GAPS` readiness verdicts.
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

1. **Check prior report**
   - Look for `./<topic-slug>-research-report.md`.
   - If present, read it for prior paper IDs, findings, confidence levels, and
     gaps. Rename it to `<topic-slug>-research-report.<date>.md` before writing
     a replacement.

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
   Before writing, load `references/output-templates.md`. Ensure the report has
   all core sections, confidence levels, evidence citations, gap
   classifications, readable diagrams, formulas when useful, and internal links.

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

If the user wants to build, design, implement, or architect a system:

1. Require a readiness verdict:
   `IMPLEMENTATION_READY`, `HAS_GAPS`, or `NOT_APPLICABLE`.
2. Classify gaps as `ACADEMIC` or `ENGINEERING`.
3. Fill engineering gaps from implementation knowledge and reliable sources.
4. Fill academic gaps with targeted new pipeline iterations.
5. Stop after readiness, no new useful gaps, or 3 iterations.

Load `references/iterative-synthesis.md` for the full loop.

## MCP Usage

The MCP server exposes the same workflow. Prefer `tool_research_workflow` for
server-driven orchestration, or use stage tools such as `tool_plan_topic`,
`tool_search`, `tool_screen_candidates`, `tool_report`, and
`tool_validate_report`. See `references/command-reference.md` for the current
tool map.

## Final Response To User

After completing a run, report:

- final report path;
- run ID and profile;
- source counts and shortlist/download/convert totals;
- top findings with confidence and citations;
- validation result and any remaining gaps;
- important artifacts such as HTML, BibTeX, or comparison outputs.
