# Command Reference

Use this file when you need exact CLI/MCP command names or source/backend
choices. In normal operation, keep only the high-level workflow in context.

## Invocation

Inside the installed skill environment, always pass:

```bash
--config ~/.claude/skills/research-pipeline/config.toml
```

`CFG` in examples means that path.

## Sources

| Source | What It Searches | Notes |
|---|---|---|
| `arxiv` | arXiv API | Default, polite rate limits |
| `scholar` | Google Scholar | Requires `scholarly` or SerpAPI |
| `semantic_scholar` | Semantic Scholar | Optional API key increases rate limits |
| `openalex` | OpenAlex | Open bibliographic metadata |
| `dblp` | DBLP | Computer science bibliography |
| `huggingface` | HuggingFace daily papers | Recent ML papers |
| `all` | All above | Best recall, slower |

Example:

```bash
research-pipeline search --run-id <RUN_ID> --source all --config CFG
```

## Core Pipeline Commands

| Step | CLI | Output |
|---|---|---|
| Plan | `research-pipeline plan "topic" --config CFG` | `plan/query_plan.json` |
| Search | `research-pipeline search --run-id ID --source all --config CFG` | `search/candidates.jsonl` |
| Screen | `research-pipeline screen --run-id ID --diversity --config CFG` | `screen/shortlist.json` |
| Download | `research-pipeline download --run-id ID --config CFG` | `download/pdf/*.pdf` |
| Convert | `research-pipeline convert --run-id ID --backend docling --config CFG` | `convert/markdown/*.md` |
| Extract | `research-pipeline extract --run-id ID --config CFG` | `extract/*.extract.json` |
| Summarize | `research-pipeline summarize --run-id ID --config CFG` | `summarize/synthesis.json` |
| Run all | `research-pipeline run "topic" --profile standard --source all --config CFG` | full run directory |
| Inspect | `research-pipeline inspect --run-id ID` | status and artifact paths |

## Profiles

| Profile | Stages |
|---|---|
| `quick` | plan, search, screen, summarize |
| `standard` | full 7-stage pipeline |
| `deep` | standard plus quality, expand, claim analysis, TER gap filling |
| `auto` | selected by query complexity |

## Conversion

Backends:

```text
docling, marker, pymupdf4llm, mineru, mathpix, datalab, llamaparse,
mistral_ocr, openai_vision
```

Commands:

```bash
research-pipeline convert --run-id <RUN_ID> --backend docling --config CFG
research-pipeline convert-file paper.pdf -o output-dir --backend pymupdf4llm --config CFG
research-pipeline convert-rough --run-id <RUN_ID> --config CFG
research-pipeline convert-fine --run-id <RUN_ID> --paper-ids "2401.12345" --backend marker --config CFG
```

Use `convert-rough` for all papers, then `convert-fine` for selected papers
when the corpus is large or when conversion cost matters.

## Expansion, Organization, And Feedback

```bash
research-pipeline quality --run-id <RUN_ID> --config CFG
research-pipeline expand --run-id <RUN_ID> --paper-ids "ID1,ID2" --direction both --config CFG
research-pipeline expand --run-id <RUN_ID> --paper-ids "ID1" --bfs-depth 2 --bfs-query "term1,term2" --config CFG
research-pipeline expand --run-id <RUN_ID> --paper-ids "ID1" --snowball --bfs-query "term1,term2" --config CFG
research-pipeline cluster --run-id <RUN_ID> --stage screen
research-pipeline enrich --run-id <RUN_ID> --stage candidates --config CFG
research-pipeline feedback --run-id <RUN_ID> --accept ID1 --reject ID2 --reason "off-topic"
```

## Claims, Evidence, And Reports

```bash
research-pipeline analyze --run-id <RUN_ID>
research-pipeline analyze --run-id <RUN_ID> --collect
research-pipeline analyze-claims --run-id <RUN_ID>
research-pipeline score-claims --run-id <RUN_ID>
research-pipeline confidence-layers --run-id <RUN_ID>
research-pipeline aggregate --run-id <RUN_ID> --min-pointers 1
research-pipeline report --run-id <RUN_ID> --template survey
research-pipeline validate --report ./<topic-slug>-research-report.md
research-pipeline export-html --markdown report.md -o report.html
research-pipeline export-bibtex --run-id <RUN_ID> --stage screen -o refs.bib
```

Report templates: `survey`, `gap_analysis`, `lit_review`, `executive`.

## Multi-Run, Memory, And Reliability Tools

```bash
research-pipeline compare --run-a <RUN_A> --run-b <RUN_B>
research-pipeline coherence <RUN_A> <RUN_B> [<RUN_C> ...]
research-pipeline consolidate [RUN_IDS...] --dry-run
research-pipeline blinding-audit --run-id <RUN_ID>
research-pipeline dual-metrics --query "topic" --run-ids r1,r2,r3
research-pipeline cbr-lookup --topic "topic"
research-pipeline cbr-retain --run-id <RUN_ID> --topic "topic" --outcome good
research-pipeline adaptive-stopping scores.json --query "topic"
```

## Knowledge Graph And Citation Context

```bash
research-pipeline kg-ingest --run-id <RUN_ID>
research-pipeline kg-stats
research-pipeline kg-query <ENTITY_ID>
research-pipeline kg-quality
research-pipeline cite-context --run-id <RUN_ID> --window 1
```

## MCP Tool Map

| Workflow Area | MCP Tools |
|---|---|
| Core stages | `tool_plan_topic`, `tool_search`, `tool_screen_candidates`, `tool_download_pdfs`, `tool_convert_pdfs`, `tool_extract_content`, `tool_summarize_papers`, `tool_run_pipeline` |
| Inspection | `tool_get_run_manifest`, `tool_list_backends`, `tool_model_routing_info`, `tool_gate_info` |
| Conversion | `tool_convert_file`, `tool_convert_rough`, `tool_convert_fine` |
| Expansion and quality | `tool_expand_citations`, `tool_evaluate_quality`, `tool_cluster`, `tool_enrich` |
| Analysis and reports | `tool_analyze_papers`, `tool_validate_report`, `tool_compare_runs`, `tool_verify_stage`, `tool_aggregate_evidence`, `tool_export_html`, `tool_export_bibtex`, `tool_report` |
| Feedback and logs | `tool_record_feedback`, `tool_query_eval_log` |
| Memory and reliability | `tool_coherence`, `tool_consolidation`, `tool_blinding_audit`, `tool_dual_metrics`, `tool_cbr_lookup`, `tool_cbr_retain` |
| KG and confidence | `tool_kg_quality`, `tool_adaptive_stopping`, `tool_confidence_layers`, `tool_cite_context`, `tool_watch` |
| Server orchestration | `tool_research_workflow` |

Prefer `tool_research_workflow` when the MCP server should orchestrate a whole
research workflow with telemetry, verification, gates, and recovery.
