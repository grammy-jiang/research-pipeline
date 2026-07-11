# Workflow Steps — research-pipeline

> **Governed by the orchestrator.** The runner (`runners/runner.py`) drives
> all tasks automatically. Load this reference only to understand what a task
> does or to diagnose a failure — not to run commands manually.
>
> Manifest task IDs are shown in brackets: `[task-id]`. Mandatory gates are
> marked **⛔ GATE** — they cannot be skipped.

## Environment setup

```bash
# Claude Code
SKILL_DIR=~/.claude/skills/research-pipeline
CFG=$SKILL_DIR/config.toml

# Copilot CLI
# SKILL_DIR=~/.copilot/skills/research-pipeline
# CFG=$SKILL_DIR/config.toml

# Codex CLI
# SKILL_DIR=~/.agents/skills/research-pipeline
# CFG=$SKILL_DIR/config.toml
```

## Task `[resume-check]` ⛔ GATE — Resume check

The runner executes this gate before every other task. It looks for
`./<topic-slug>-research-report.md` in the CWD.

If found: snapshot-renames the old file and writes `resume_context.json`
with `prior_paper_ids` and `open_gaps_raw` to seed the new run.

```bash
# Runner invokes this automatically. Direct invocation (debug only):
bash "$SKILL_DIR/scripts/resume-check.sh" "<topic-slug>" "$(pwd)"
```

Manual inspection of the context JSON:

```python
import json
ctx = json.load(open("resume_context.json"))
# ctx["resume"]           — bool (True if a prior report was found)
# ctx["snapshot"]         — snapshot path of the renamed prior report
# ctx["prior_paper_ids"]  — IDs to feed into expand --paper-ids
# ctx["open_gaps_raw"]    — open gap lines to seed new query_variants
```

## Tasks `[plan]` ⛔ GATE + `[verify-plan]` ⛔ GATE — Plan

```bash
research-pipeline plan "<topic>" --config "$CFG"
research-pipeline verify --run-id <RUN_ID> --stage plan --config "$CFG"
```

Review `runs/<run_id>/plan/query_plan.json`:
- Keep `must_terms` to 2-3 terms.
- Add synonym-rich `query_variants`.
- Consult `references/query-optimization.md` when recall matters.

## Tasks `[search]` + `[paper-screener]` — Search and screen

```bash
research-pipeline search --run-id <RUN_ID> --source all --config "$CFG"
research-pipeline screen --run-id <RUN_ID> --diversity --config "$CFG"
```

`--source all` searches arXiv, Scholar, Semantic Scholar, OpenAlex, DBLP, and
HuggingFace daily papers. The runner delegates `[paper-screener]` to the
`paper-screener` sub-agent when BM25 results are noisy or the topic is broad.
See `references/sub-agents.md` for delegation conditions.

## Tasks `[quality]` + `[expand]` + `[enrich]` — Optional expansion (standard/deep profiles)

```bash
research-pipeline quality --run-id <RUN_ID> --config "$CFG"
research-pipeline expand --run-id <RUN_ID> --paper-ids "ID1,ID2" --direction both --config "$CFG"
research-pipeline enrich --run-id <RUN_ID> --config "$CFG"
```

- `[expand]`: citation graph recall; required when resuming with prior paper IDs.
- `[quality]`: citation/venue/author signals; useful for deep profiles.
- `[enrich]`: fill missing metadata from Semantic Scholar.

## Tasks `[download]` + `[convert-rough]` + `[convert-fine]` + `[extract]` — Download and convert

```bash
research-pipeline download --run-id <RUN_ID> --config "$CFG"
research-pipeline convert-rough --run-id <RUN_ID> --config "$CFG"
research-pipeline convert-fine --run-id <RUN_ID> --paper-ids "ID1,ID2" --config "$CFG"
research-pipeline extract --run-id <RUN_ID> --config "$CFG"
```

For small runs (< 10 papers), `research-pipeline convert` is sufficient.
For larger runs: rough-convert all PDFs, then fine-convert the important subset.

## Tasks `[summarize]` + `[analyze-claims]` + `[score-claims]` — Summarize and analyze

```bash
research-pipeline summarize --run-id <RUN_ID> --config "$CFG"
research-pipeline analyze-claims --run-id <RUN_ID>
research-pipeline score-claims --run-id <RUN_ID>
```

`summarize` writes:
- `summarize/extractions/*.extraction.json` — typed statements, evidence, confidence
- `summarize/synthesis_report.json` — structured cross-paper synthesis
- `summarize/synthesis_traceability.json` — evidence lineage

## Tasks `[paper-analyzer]` + `[paper-synthesizer]` + `[review-synthesis]` — Sub-agent analysis (deep profile)

The runner delegates these tasks to sub-agents with formal contracts:

1. `[paper-analyzer]` → `paper_analyzer` sub-agent (one per paper, parallel)
2. `[paper-synthesizer]` → `paper_synthesizer` sub-agent
3. `[review-synthesis]` → `synthesis-reviewer` sub-agent (llm_reviewer gate)

See `runners/subagent_contracts/` for the full contracts.
Sub-agents inherit the session model by default (recommended); prefer a model
alias (`opus`/`sonnet`/`haiku`) over a pinned dated id, and only override
downward for mechanical steps. See `references/sub-agents.md` for details.

## Task `[classify-gaps]` — Gap classification

```bash
# Runner delegates to gap_classifier sub-agent (llm_worker_task).
# There is no standalone CLI command for this task.
# To debug: re-run runner.py and inspect the delegation output for the
# gap_classifier contract, or examine {cwd}/gaps.json after the round.
```

Reads `synthesis_report.json` open_gaps (and optionally `{run_dir}/analysis/synthesis.json`
for deep profile). Writes `{cwd}/gaps.json` with each gap typed as `ACADEMIC`
(requires new papers) or `ENGINEERING` (fillable from docs/code).

## Task `[report]` — Write final report

```bash
research-pipeline report --run-id <RUN_ID> --template structured_synthesis --config "$CFG"
```

Then the report is written to `./<topic-slug>-research-report.md`. Required sections:

- `## Contents` — table of contents with internal Markdown links
- `## Round History` — per-round table (see `references/output-templates.md`)
- `## Executive Summary`, `## Research Question`, `## Methodology`, `## Papers Reviewed`
- `## Research Landscape`, `## Confidence-Graded Findings`, `## Research Gaps`
- `## Practical Recommendations`, `## Evidence Map`, `## References`

Required formatting:
- Confidence annotations: `[HIGH]`, `[MEDIUM]`, `[LOW]`
- Mermaid diagrams: `flowchart TD`/`TB` (never ASCII art)
- LaTeX formulas: `$...$` inline, `$$...$$` display
- Gap labels: `[ACADEMIC]` / `[ENGINEERING]` in Research Gaps

## Task `[validate-report]` ⛔ GATE — Validate

```bash
research-pipeline validate --report "./<topic-slug>-research-report.md" --config "$CFG"
research-pipeline export-html --markdown "./<topic-slug>-research-report.md" \
    -o "./<topic-slug>-research-report.html" --config "$CFG"
research-pipeline export-bibtex --run-id <RUN_ID> --stage screen -o refs.bib
```

The runner blocks on this gate. Validation failures must be fixed before
`[check-completion]` can run.

## Task `[check-completion]` ⛔ GATE — Completion gate

```bash
# Runner invokes this automatically. Direct invocation (debug only):
python3 "$SKILL_DIR/scripts/check_completion.py" --run-id <RUN_ID> --slug "<topic-slug>"
```

Exit 1: fix the reported issues before delivering to the user.

## Gap-closure rounds (iterative synthesis)

After `[check-completion]` passes, read `gaps.json`. If
`convergence.should_continue: true` and the round cap has not been reached:

1. Take ACADEMIC gaps → new search queries → start a new round (re-run runner).
2. Take ENGINEERING gaps → fill from documentation/code knowledge.
3. Stop when: no open gaps, round 4 complete, zero new papers found, or user
   marks remaining gaps out-of-scope.

See `references/iterative-synthesis.md` for the full protocol.

## Optional quality diagnostics (deep profile)

```bash
# A3-5 Unified Horizon Metric
research-pipeline horizon --score 0.8 --achieved 40 --target 50

# Theme 16 Recall/Reasoning/Presentation diagnostic
research-pipeline rrp --report report.md --shortlist shortlist.json
```

If `rrp` reports `recall` as the bottleneck, rerun screening or expand the
candidate set before rewriting prose.

## Profiles

| Profile | Manifest task set | Use when |
|---------|------------------|----------|
| `quick` | 12 tasks | Fast abstract-only overview |
| `standard` | 17 tasks | Normal evidence-backed review |
| `deep` | 23 tasks | Comprehensive review with quality, expansion, claim analysis |
| `auto` | runner decides | Let the CLI choose based on query complexity |

```bash
python3 $SKILL_DIR/runners/runner.py "<topic>" --profile standard --config "$CFG"
python3 $SKILL_DIR/runners/runner.py "<topic>" --profile deep    --config "$CFG"
```
