---
name: research-pipeline
description: >
  End-to-end academic paper research workflow: plan search queries, retrieve
  candidates from arXiv, Google Scholar, and HuggingFace daily papers, screen by relevance (BM25 +
  optional SPECTER2 re-ranking), evaluate paper quality, expand via citation
  graph, download PDFs, convert to Markdown (docling, marker, pymupdf4llm),
  extract content, and generate evidence-backed summaries. For system-building
  goals, performs iterative synthesis with gap analysis until
  implementation-ready. Uses the research-pipeline CLI. Use when user says
  "search arXiv", "find papers on", "research papers about", "literature
  review", "paper survey", "screen papers", "summarize papers", "search
  Google Scholar", "citation graph", "paper quality", or asks to find,
  download, or analyze academic papers. Do NOT use for general web search
  or simple PDF reading (use pdf-to-markdown skill instead).
metadata:
  author: grammy-jiang
  version: 1.4.0
  category: research
  tags: [arxiv, scholar, papers, research, literature-review, academic, citation-graph, quality-evaluation]
---

# Academic Paper Research

End-to-end academic paper research pipeline producing auditable, reproducible
artifacts. Searches arXiv, Google Scholar, and HuggingFace daily papers; screens relevance; evaluates
quality; expands via citation graph; downloads PDFs; converts to Markdown;
and generates evidence-backed summaries with sub-agent analysis.

## Prerequisites

| Requirement | Install Command | When Needed |
|-------------|----------------|-------------|
| CLI tool | `pipx install research-pipeline` | Always |
| Docling backend | `pipx inject research-pipeline docling` | Convert (MIT, good tables/equations) |
| Marker backend | `pipx inject research-pipeline marker-pdf` | Convert (best accuracy 95.7%, GPL-3.0) |
| PyMuPDF4LLM backend | `pipx inject research-pipeline pymupdf4llm` | Convert (fastest 10-50x, AGPL) |
| MinerU backend | `pipx inject research-pipeline magic-pdf` | Convert (TEDS 93.42% tables, MIT) |
| Scholar (free) | `pipx inject research-pipeline scholarly` | `--source scholar` or `--source all` |
| Scholar (paid) | `pipx inject research-pipeline google-search-results` + set `RESEARCH_PIPELINE_SERPAPI_KEY` | Production Scholar |

## CLI Invocation

**IMPORTANT**: `research-pipeline` is installed via **pipx** and available
globally. Always call it **directly** — never prefix with `uv run`, `python -m`,
or any other wrapper:

```bash
# CORRECT — call directly
research-pipeline plan "topic" --config ~/.claude/skills/research-pipeline/config.toml

# WRONG — do NOT use uv run (only for development in the source repo)
# uv run research-pipeline plan "topic" --config ...
```

**Every CLI invocation MUST pass `--config ~/.claude/skills/research-pipeline/config.toml`**
to load settings regardless of CWD. Abbreviated as `CFG` below.

## Pipeline Overview

```mermaid
flowchart LR
    plan --> search --> screen
    screen --> quality["[quality]"]
    quality --> expand["[expand]"]
    expand --> download --> convert --> extract --> summarize
    summarize -.-> analyze-claims["[analyze-claims]"]
    analyze-claims -.-> score-claims["[score-claims]"]
    convert -.-> convert-rough["[convert-rough]"]
    convert -.-> convert-fine["[convert-fine]"]
```

7 core stages plus optional quality scoring, citation expansion, and tiered
conversion. After the pipeline, sub-agents (paper-screener, paper-analyzer,
paper-synthesizer) provide intelligent analysis. For system-building goals,
iterative synthesis fills gaps until implementation-ready.

## Command Reference

All commands accept `--config CFG` (required) and `--verbose` (optional).
Stage commands require `--run-id ID`.

| # | Stage | Command | Key Options |
|---|-------|---------|-------------|
| 1 | Plan | `research-pipeline plan "topic" --config CFG` | `--run-id ID` (optional, auto-generated) |
| 2 | Search | `research-pipeline search --run-id ID --config CFG` | `--source arxiv\|scholar\|huggingface\|all` |
| 3 | Screen | `research-pipeline screen --run-id ID --config CFG` | `--resume` |
| 3b | Quality | `research-pipeline quality --run-id ID --config CFG` | _(optional stage)_ |
| 3c | Expand | `research-pipeline expand --run-id ID --paper-ids "ID1,ID2" --config CFG` | `--direction both\|citations\|references`, `--limit N`, `--reference-boost F`, `--bfs-depth D`, `--bfs-top-k K`, `--bfs-query "term1,term2"` |
| 4 | Download | `research-pipeline download --run-id ID --config CFG` | `--force` |
| 5 | Convert | `research-pipeline convert --run-id ID --config CFG` | `--backend docling\|marker\|pymupdf4llm\|mineru`, `--force` |
| 5b | Rough | `research-pipeline convert-rough --run-id ID --config CFG` | `--force` |
| 5c | Fine | `research-pipeline convert-fine --run-id ID --paper-ids "ID1,ID2" --config CFG` | `--backend B`, `--force` |
| 6 | Extract | `research-pipeline extract --run-id ID --config CFG` | — |
| 7 | Summarize | `research-pipeline summarize --run-id ID --config CFG` | — |
| 8 | Analyze Claims | `research-pipeline analyze-claims --run-id ID` | — |
| 9 | Score Claims | `research-pipeline score-claims --run-id ID` | — |
| All | Run | `research-pipeline run "topic" --config CFG` | `--source all` |
| — | Inspect | `research-pipeline inspect --run-id ID` | `--workspace PATH` |
| — | Convert File | `research-pipeline convert-file path.pdf --config CFG` | `-o DIR`, `--backend B` |
| — | Index | `research-pipeline index --list` | `--gc`, `--search QUERY`, `--search-limit N` |

All run artifacts stored under `runs/<run_id>/`.

## Step-by-Step Workflow

### Step 0: Check for Existing Report

Before starting, look for `<topic-slug>-research-report.md` in the CWD.
If found:

1. **Read it as prior context** — extract already-analyzed paper IDs, prior
   findings, confidence levels, and gap classifications.
2. **Do NOT append-merge** into the old report. Appending produces duplicated
   findings, stale conclusions, and patchwork reports.
3. **Generate a fresh report** from the current evidence state after all
   pipeline stages and sub-agent analysis complete.
4. Optionally include a **Prior Run Comparison** section in the new report:
   ```markdown
   ## Prior Run Comparison
   - Previous report: <filename>
   - Newly added papers: ...
   - Confidence changes: ...
   - Gaps resolved: ...
   - Remaining gaps: ...
   ```
5. The old report becomes a historical artifact — rename it to
   `<topic-slug>-research-report.<date>.md` before writing the new one.

### Step 1: Plan

```bash
research-pipeline plan "multimodal RAG for long-document QA" --config CFG
```

Output: `runs/<run_id>/plan/query_plan.json` with normalized topic,
must/nice/negative terms, time window (default: 6 months).
Report the run ID and parsed search terms to the user.

### Step 1.5: Validate the Query Plan (RECOMMENDED)

Since v0.4.0, the plan stage auto-generates reasonable defaults:
- **Stop words** are filtered (e.g., "with", "for", "the")
- **must_terms** capped at 3, overflow goes to **nice_terms**
- **query_variants** auto-generated (up to `max_query_variants` from config)

Use the verification gate to confirm structural validity:
```bash
research-pipeline verify --run-id <RUN_ID> --stage plan
```
This checks that `must_terms` (≥1) and `query_variants` (≥2) are present.

You should still review and refine:

1. **Verify `must_terms`** — ensure they capture the core concepts
2. **Cap `must_terms` to 2** if queries are still too narrow
3. **Expand query_variants** — add synonym-expanded variants using different
   vocabulary for each key concept (the auto-generated ones use term
   reordering; add vocabulary substitutions manually)
4. Include a "core-concept-only" variant without domain qualifiers
5. Cross-check: would a paper using different terminology still match?
6. Set `primary_months` to the user's requested time window

Write the improved plan back to `query_plan.json`.
Consult `references/query-optimization.md` for synonym tables and examples.

### Step 2: Search

```bash
research-pipeline search --run-id <RUN_ID> --source all --config CFG
```

`--source all` searches **arXiv + Google Scholar + HuggingFace daily papers**
in parallel. Results are deduplicated by arXiv ID and normalized title.

Available source values for `--source`:

| Value | Searches | Notes |
|-------|----------|-------|
| `arxiv` | arXiv API | Default source |
| `scholar` | Google Scholar | Requires `scholarly` or SerpAPI |
| `huggingface` | HuggingFace daily papers | Keyword-filtered, recent papers |
| `all` | arXiv + Scholar + HuggingFace | All searchable sources in parallel |

**Note**: Semantic Scholar, OpenAlex, and DBLP are used by the `expand`
(citation graph) and `quality` (author h-index) commands but are **not**
searchable via `--source`.

HuggingFace configuration in `config.toml`:
```toml
[sources]
huggingface_enabled = true
huggingface_min_interval = 0.5  # seconds between requests
huggingface_limit = 100         # max daily papers to fetch
```

Report: candidates per source, total unique after dedup, run ID.

### Step 3: Screen

```bash
research-pipeline screen --run-id <RUN_ID> --config CFG
```

BM25 title/abstract scoring + category matching + recency bonus.
Content is automatically sanitized (prompt injection patterns stripped)
before scoring. Output includes `depth_tiers.json` with confidence-gated
retrieval depth classification (high/medium/low) for adaptive expansion.
Report: total scored, shortlist count, top 5 papers with scores.

**Query refinement feedback** (v0.12.0+): After screening, the pipeline
automatically analyzes term overlap between the query and shortlisted
papers. Results are saved to `query_refinement.json` with:
- `suggested_additions`: domain terms frequent in top-K but absent from query
- `suggested_removals`: query terms with low coverage in results
- `emergent_terms`: newly discovered terms from the corpus
- `term_coverage`: per-term hit rate across shortlisted papers
Use this feedback to refine the query plan in iterative searches.

**Diversity-aware shortlisting** (v0.11.0+): Use `--diversity` to enable
MMR-style reranking that balances relevance with coverage of different
categories, time periods, and sources. Prevents echo-chamber shortlists.
```bash
research-pipeline screen --run-id <RUN_ID> --diversity --diversity-lambda 0.3 --config CFG
```
`--diversity-lambda` controls balance: 0.0 = pure relevance, 1.0 = pure
diversity, 0.3 (default) gives moderate diversity.

**LLM second-pass screening** (v0.11.0+): When LLM is configured
(`[llm] enabled = true`), the screen stage automatically runs a second
pass using the LLM judge on shortlisted papers. The final score becomes
a 60/40 blend of heuristic and LLM scores. When LLM is not configured,
screening uses heuristic-only mode (default).

**Known limitation**: BM25 screening can be noisy with broad topic terms
(e.g., "agent" + "evaluation"). If the shortlist contains many irrelevant
papers, use the paper-screener sub-agent for intelligent re-screening.

### Step 3.5: Quality Evaluation (Optional)

```bash
research-pipeline quality --run-id <RUN_ID> --config CFG
```

Composite quality scoring: citation impact, venue reputation, author
credibility, and recency bonus. Papers flagged as retracted or fabricated
(safety_flag) are zeroed automatically — safety acts as a multiplicative
gate. Output: `quality/quality_scores.jsonl`.

### Step 3.6: Citation Graph Expansion (Optional)

```bash
research-pipeline expand --run-id <RUN_ID> --paper-ids "2401.12345,2402.67890" --direction both --limit 20 --config CFG
```

**`--paper-ids` is required.** Expand the candidate pool by traversing the
citation graph via Semantic Scholar. `--direction citations` finds papers
citing yours; `references` finds their references; `both` does both.

**Backward-preference**: Use `--reference-boost 2.0` when `--direction both`
to fetch 2× more references than citations. Research shows backward references
yield +10-20 pp higher recall because foundational papers curate high-quality
reference lists. Default 1.0 (equal treatment).

**BFS multi-hop expansion** (v0.10.0+): Use `--bfs-depth 2` to perform
breadth-first traversal with BM25 pruning at each hop (+24pp recall).
Requires `--bfs-query "term1,term2"` for relevance pruning. Example:
```bash
research-pipeline expand --run-id <RUN_ID> --paper-ids "2401.12345" \
  --bfs-depth 2 --bfs-top-k 10 --bfs-query "transformer,attention" --config CFG
```

### Step 4: Download

```bash
research-pipeline download --run-id <RUN_ID> --config CFG
```

Rate-limited, SHA-256 verified, idempotent. Report: count, size, failures.

The download stage uses **lenient shortlist parsing** — you can safely edit
`shortlist.json` to add or remove papers without providing every field.
A minimal entry needs only the `paper` object (with `arxiv_id`, `version`,
`title`, `authors`, `published`, `updated`, `categories`, `primary_category`,
`abstract`, `abs_url`, `pdf_url`), plus `download: true`. Missing fields
like `cheap`, `download_reason`, and `final_score` will be auto-filled
with sensible defaults.

### Step 5: Convert

```bash
research-pipeline convert --run-id <RUN_ID> --config CFG
```

Default backend: `docling`. Override with `--backend marker`, `--backend pymupdf4llm`, or `--backend mineru`.

**Two-tier conversion** (recommended for 10+ papers):

```bash
# Fast rough conversion of ALL PDFs
research-pipeline convert-rough --run-id <RUN_ID> --config CFG

# High-quality conversion of SELECTED papers only
research-pipeline convert-fine --run-id <RUN_ID> --paper-ids "2401.12345,2402.67890" --config CFG
```

Since v0.4.0, the extract stage **auto-merges** two-tier manifests. You no
longer need to manually create a unified `convert/` directory — just run
`extract` directly after `convert-rough` and/or `convert-fine`. Fine-tier
entries override rough-tier for the same paper.

### Step 6: Extract

```bash
research-pipeline extract --run-id <RUN_ID> --config CFG
```

Produces per-paper extraction files (`{arxiv_id}{version}.extract.json`) and
bibliography files (`{arxiv_id}{version}.bibliography.json`) in
`runs/<run_id>/extract/`. Bibliography entries contain parsed arXiv IDs and
DOIs that can seed citation graph expansion without the Semantic Scholar API.

**Cross-encoder reranking** (v0.12.1+): When `sentence-transformers` is installed,
chunk retrieval automatically applies cross-encoder reranking on the top BM25
candidates for higher precision. Use `--cross-encoder` to force enable/disable.

### Step 7: Summarize

Per-paper evidence-driven summarization + cross-paper synthesis.

**LLM-powered summarization** (v0.11.0+): When LLM is configured
(`[llm] enabled = true`), summarization uses the LLM provider for
structured per-paper summaries (objective, methodology, findings,
limitations) and cross-paper synthesis (agreements, disagreements,
open questions). When LLM is not configured, uses template mode
that extracts relevant sections and aggregates findings heuristically.

**Export formats** (v0.12.0+): Use `--output-format` to produce
additional output alongside the default synthesis JSON:
```bash
research-pipeline summarize --run-id <RUN_ID> -f json --config CFG
research-pipeline summarize --run-id <RUN_ID> -f bibtex --config CFG
research-pipeline summarize --run-id <RUN_ID> -f structured-json --config CFG
```
- `json`: full synthesis with metadata + evidence map
- `bibtex`: bibliography entries for all papers (paste into LaTeX)
- `structured-json`: flat claim list with explicit evidence chains
  (claim → paper → chunk → quote) and confidence levels

```bash
research-pipeline summarize --run-id <RUN_ID> --config CFG
```

Report: papers summarized, key findings.

### Step 8: Claim Decomposition (v0.12.3+)

**Claim decomposition**: After summarization, run `analyze-claims` to
decompose findings into atomic claims with evidence classification (supported/
partial/conflicting/inconclusive/unsupported). This helps identify which claims
are well-supported by source text.

```bash
research-pipeline analyze-claims --run-id <RUN_ID>
```

Report: claims per paper, evidence class distribution.

### Step 9: Confidence Scoring (v0.12.5+)

**Confidence scoring**: After claim decomposition, run `score-claims` to
compute multi-signal confidence scores per claim. Signals include evidence
strength, hedging language detection (LVU), citation density, and retrieval
quality. When an LLM is available, adds multi-sample consistency verification.

```bash
research-pipeline score-claims --run-id <RUN_ID>
```

Report: papers scored, total claims, average confidence.

### Step 10: Knowledge Graph (v0.12.4+)

**Knowledge graph**: The pipeline builds a typed knowledge graph in SQLite
tracking papers, concepts, methods, claims, and their relationships. Use
`kg-ingest` after summarization to populate, `kg-stats` to view, and
`kg-query <entity-id>` to explore relationships.

```bash
research-pipeline kg-ingest --run-id <RUN_ID>
research-pipeline kg-stats
research-pipeline kg-query 2401.12345
```

Report: entities and triples ingested, with provenance tracking.

## Sub-Agent Analysis

After the pipeline completes, use three specialized sub-agents for intelligent
paper evaluation. Launch them via the task tool with the appropriate agent type.

**Model Requirement — Preferred-Model Policy**: All sub-agents SHOULD be
launched with the **strongest available reasoning model** for maximum quality.
Academic paper analysis demands high-quality reasoning — do not use fast or
cheap models for sub-agents.

**Model tiers**:
| Tier | Policy | Examples |
|------|--------|---------|
| Preferred | Strongest reasoning model available | `claude-opus-4.6`, `claude-opus-4.5` |
| Fallback | Approved secondary reasoning model | `claude-sonnet-4.5`, `gpt-4.1` |
| Minimum | Do not run synthesis below this tier | Any model weaker than the fallback tier |

If the preferred model is unavailable, use the best fallback and **annotate
reduced confidence** in the synthesis output. Log which model was actually
used in the run metadata (`subagent_model_used` field).

```python
# Example: launching a paper-analyzer sub-agent
task(
    agent_type="paper-analyzer",
    model="claude-opus-4.6",      # ← Preferred; agent will use best available
    mode="background",
    name="az-paper-name",
    prompt="...",
)
```

### Typical flow

```mermaid
flowchart TD
    A["1. Run pipeline stages<br/>(plan through summarize)"] --> B
    B["2. paper-screener<br/>intelligent screening beyond BM25"] --> C
    C["3. paper-analyzer × N<br/>deep per-paper analysis (parallel)"] --> D
    D["4. paper-synthesizer<br/>cross-paper synthesis + gap detection"] --> E
    E["5. Write final report to CWD"]

    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
```

All sub-agents SHOULD use the strongest available reasoning model
(see model tiers above).

Consult `references/sub-agents.md` for prompt templates, I/O specs, and
model configuration details.

### paper-screener

When to use: After screening, if BM25 shortlist quality is uncertain.

Provide: run directory path, run ID, research topic, and any focus areas
or exclusion criteria. Agent reads `search/candidates.jsonl` or
`screen/cheap_scores.jsonl` and returns: shortlist count, top papers,
coverage gaps. **Launch with the strongest reasoning model available.**

### paper-analyzer

When to use: After convert stage, for detailed individual paper analysis.

Launch **one agent per paper** in parallel for efficiency. Provide: path to
the paper's Markdown file, research topic, and analysis focus (methodology,
scalability, etc.). Agent returns: rating (1-5 stars), methodology assessment,
key findings, transferable patterns, limitations. **Launch with
the strongest reasoning model available.**

### paper-synthesizer

When to use: After all paper-analyzer agents complete.

Provide: all analysis summaries, research topic, and whether this is
system-building mode. Agent returns: themes, contradictions, gaps,
recommendations, and readiness assessment. **Launch with the strongest
reasoning model available.**

## Output Requirements

After each stage, report status to the user with a pipeline status table.
The final report MUST be written to the **CWD** (not `runs/`):

```
./<topic-slug>-research-report.md
```

### Formatting Rules

- Use **Mermaid** for all diagrams and flowcharts (never ASCII art)
- Use **LaTeX** for mathematical formulas: inline `$...$`, display `$$...$$`
- Use **Markdown tables** for structured comparisons

### Required Report Sections

The final report MUST include all **core sections** and should include
**conditional sections** when evidence justifies them (see
`references/output-templates.md` for the full template and trigger logic):

**Core sections** (always required):
1. **Executive Summary** — scope, confidence level, verdict
2. **Research Question** — precise scope boundaries
3. **Methodology** — search strategy, pipeline summary with Mermaid diagram
4. **Papers Reviewed** — table with quality scores and relevance
5. **Research Landscape** — themes with per-theme confidence and citations
6. **Research Gaps** — classified as ACADEMIC vs ENGINEERING with severity
7. **Practical Recommendations** — evidence-backed, with confidence level
8. **References** — full reference list
9. **Appendix: Run Metadata** — run ID, sources, version, date

**Conditional sections** (include when justified):
- **Methodology Comparison** — when 2+ distinct approaches are studied
- **Confidence-Graded Findings** — 🟢 High / 🟡 Medium / 🔴 Low with evidence
- **Trade-Off Analysis** — when real alternative approaches exist
- **Points of Agreement** — when 2+ papers materially agree
- **Points of Contradiction** — when 2+ papers materially disagree
- **Reproducibility Notes** — when code/data availability is relevant
- **Evidence Map** — for 5+ papers or when auditability is required
- **Readiness Assessment** — in system-building mode only
- **Future Directions** — when novel research directions emerge from the literature

### Confidence Level Rules (Heuristic Guidelines)

- **🟢 High**: 3+ papers, consistent methodology, reproducible results
- **🟡 Medium**: 1-2 papers, or consistent results with methodology caveats
- **🔴 Low**: single source, contradicted by other papers, or preliminary/unreproduced

These thresholds are operational heuristics, not epistemic guarantees.
Every finding, recommendation, and theme MUST include its confidence level.

### Report Validation

After writing the final report, validate it with:
```bash
research-pipeline validate --report ./<topic-slug>-research-report.md
```
This checks all 9 core sections, confidence annotations, evidence
citations, gap classifications, tables, Mermaid diagrams, and LaTeX formulas.
Conditional sections earn bonus credit when present.
The verdict is PASS (score ≥ 0.7 with all core sections present) or FAIL.

**RACE quality scoring** (v0.11.0+): Validation now includes RACE
dimension scores in the output:
- **R**eadability: sentence length, paragraph structure, heading density
- **A**ctionability: actionable recommendations, specific findings
- **C**omprehensiveness: section coverage, word count, citation diversity
- **E**vidence: citation density, evidence quotes, confidence annotations

The RACE overall score (0-1) appears in `validation_result.json` under
`race_score`. Use it to identify weak areas in the report.

**FACT citation verification** (v0.12.0+): When validating with `--run-id`,
the pipeline loads paper IDs from the run's shortlist and verifies that
citations in the report reference real papers in the corpus:
```bash
research-pipeline validate --run-id <RUN_ID> --workspace runs
```
FACT metrics in `validation_result.json` under `fact_score`:
- `citation_accuracy`: fraction of citations pointing to known papers
- `effective_citation_ratio`: fraction of corpus papers actually cited
- `unsupported_citations`: citation refs not matching any paper
- `uncited_papers`: papers in corpus never cited in the report

### Cross-Run Comparison

When running iterative synthesis, compare runs to track progress:
```bash
research-pipeline compare --run-a <PREV_RUN_ID> --run-b <NEW_RUN_ID>
```
This produces a structured diff: papers added/removed, gaps resolved/new,
confidence-level changes, and readiness progression.

**Enhanced comparison** (v0.12.0+): Comparison now includes:
- **Jaccard similarity**: overlap coefficient for paper sets
- **Quality summary**: mean/median/min/max composite scores per run
- **Topic drift detection**: TF-based analysis of title/abstract term shifts
- **Source distribution**: papers per source (arXiv, Scholar, S2, etc.)
  between runs to show search strategy evolution

## System-Building Mode

When the user's goal is to **build a system** (keywords: "build", "implement",
"design", "create", "develop", "architecture", "system"), the synthesis must
be evaluated for implementation-readiness:

- **Engineering gaps**: fill with own knowledge + web research
- **Academic gaps**: run new pipeline iterations with gap-specific queries
- **Max 3 iterations**, each narrowing the search
- Stop when synthesizer returns `IMPLEMENTATION_READY`, `NOT_APPLICABLE`, or no new gaps

After the final report, ask whether to hand over to `req-analysis` skill.
Consult `references/iterative-synthesis.md` for the full protocol.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `command not found: research-pipeline` | Not installed | `pipx install research-pipeline` |
| No candidates found | Query too narrow | Broaden terms, use `--source all`, expand time window |
| Shortlist mostly irrelevant | Broad must_terms | Use paper-screener sub-agent; refine must_terms to 2 |
| Conversion fails (docling) | Missing dependency | `pipx inject research-pipeline docling` |
| Scholar SKIPPED on `--source all` | scholarly not installed | `pipx inject research-pipeline scholarly` |
| 429 rate limit | Too many API calls | Automatic retry with backoff; wait and retry |
| PDF download fails | arXiv rate limit | Pipeline retries automatically; use `--force` to re-try |
| Extract fails after two-tier convert | Pipeline < v0.4.0 | Upgrade: `pipx install --force research-pipeline` |
| ValidationError on download | Manually-edited shortlist | v0.4.0+ uses lenient parsing; upgrade if needed |
| Empty `logs/` directory | Pipeline < v0.3.1 | Upgrade: `pipx install --force research-pipeline` |

Consult `references/troubleshooting.md` for full error table, rate limits,
caching, and configuration details.

## Logging

Since v0.3.1, all pipeline runs write structured JSONL logs to
`runs/<run_id>/logs/pipeline.jsonl`. These are useful for debugging
failures and auditing pipeline behavior. Use `--verbose` for DEBUG-level
output.

## Version History

| Version | Date | Type | Summary |
|---------|------|------|---------|
| v1.3.0 | 2026-04-16 | Feature | LLM provider abstraction (Ollama + OpenAI-compatible), LLM screening judge, LLM per-paper summarization, LLM cross-paper synthesis, diversity-aware shortlisting (`--diversity`), RACE report quality scoring |
| v1.2.0 | 2026-04-15 | Feature | Safety-as-multiplicative-gate, BFS citation expansion (`--bfs-depth`), content sanitization, confidence-gated retrieval depth, enhanced checkpoint/resume, hybrid BM25+SPECTER2 retrieval |
| v1.1.0 | 2026-04-14 | Feature | Backward-preference citation (`--reference-boost`), Q2D query augmentation, FTS5 index search (`--search`), bibliography extraction in extract stage, audit logging, tool integrity hashing |
| v1.0.3 | 2026-04-15 | Quality | Release polish: engineering-gap regenerate-in-place, extract filename example, search-vs-supporting-API separation, screener input label |
| v1.0.2 | 2026-04-15 | Quality | Final cleanup: version alignment, extract artifact naming ({arxiv_id}{version}.extract.json), quality_scores.jsonl extension fix, NOT_APPLICABLE in all readiness references |
| v1.0.1 | 2026-04-15 | Quality | Final consistency: unified synthesis→summarize/ paths, HuggingFace in all examples, NOT_APPLICABLE in final message contract, confidence heuristic qualifier |
| v1.0.0 | 2026-04-15 | Quality | Cross-file consistency: unified source contract, report policy, artifact paths, schema governance, execution vocabulary; fixed shortlist.json extension bugs in quality/compare commands |
| v0.9.0 | 2026-04-14 | Feature | Modular report template; preferred-model policy; regenerate-from-evidence; HuggingFace source; expanded query optimization; citation-granularity policy; license awareness |
| v0.7.1 | 2026-03-15 | Feature | Enhanced report template: methodology comparison, confidence-graded findings, trade-off analysis, gap classification, reproducibility notes; structured agent output schemas |
| v0.4.0 | 2026-02-01 | Feature | Auto-merge two-tier converts; auto-generate query_variants; lenient shortlist parsing; scholarly installed; preferred-model policy for sub-agents |
| v0.3.1 | 2026-01-15 | Fix | Per-run file logging; version detection fixes; write_jsonl arg order fix |
| v0.3.0 | 2026-01-01 | Feature | Multi-source search; quality scoring; citation graph expansion; cloud conversion backends |

## Key Constraints

- **Rate limits**: arXiv 1 req/3s (never parallel); Scholar 10s+
- **Query terms**: Cap `must_terms` at 2–3 for recall
- **Synonyms**: Always generate query variants with different vocabulary
- **Evidence**: Every summary claim must cite source (paper_id, section)
- **Cache**: `~/.cache/research-pipeline/` (PDF + Markdown, 6 months)
- **429 retry**: Automatic exponential backoff with jitter
