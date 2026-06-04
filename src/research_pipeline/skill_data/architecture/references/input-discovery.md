# Reference: Blueprint Input Discovery and Topic-Slug Derivation

Load when resolving which blueprint to use (prompt 01) or naming the output
(prompt 24).

## Input discovery priority

```text
1. Current conversation context:
   - Use the most recently generated or discussed product blueprint if visible.
2. Current working directory, search for:
   - *-product-blueprint.md
   - *blueprint*.md
   - product-blueprint.md
3. Common project directories:
   project-design/  design/  docs/  docs/design/  output/  artifacts/
4. If multiple candidates exist:
   - Prefer the most recently modified file.
   - Prefer filenames containing "product-blueprint".
   - Prefer files containing the blueprint marker sections (below).
5. If no candidate is found:
   - Stop and ask the user for the blueprint path.
```

## Blueprint candidate scoring

| Signal | Score |
|---|---:|
| Filename contains `product-blueprint` | +5 |
| Filename contains `blueprint` | +3 |
| Contains `# Product Blueprint` | +5 |
| Contains `Executive Product Thesis` | +3 |
| Contains `MVP Scope` | +3 |
| Contains `Logical Architecture` | +3 |
| Contains `Handoff Notes for Technical Design` | +4 |
| Contains `Traceability Appendix` | +2 |
| Recently modified | +1 to +3 |

Pick the highest-scoring candidate. If the top candidates are close, ask the
user in interactive mode; otherwise proceed automatically and record the
assumption.

## Topic-slug derivation

```text
1. If the filename matches <topic-slug>-product-blueprint.md → use <topic-slug>.
2. If the filename matches <topic-slug>-blueprint.md → use <topic-slug>.
3. Otherwise derive a lowercase hyphenated slug from the product name in the
   blueprint title.
4. If still ambiguous, use architecture-design.md and record the naming
   assumption.
```

Examples:

```text
llm-agent-translation-system-product-blueprint.md
  → llm-agent-translation-system-architecture-design.md
ai-memory-system-blueprint.md
  → ai-memory-system-architecture-design.md
```

The architecture output is co-located with the source blueprint unless the
user specifies another output directory.

## Context budget / blueprint compaction

Create a compacted extract (`intermediate/blueprint_architecture_extract.md`)
only if the blueprint is too large for reliable full-context reasoning, exceeds
the configured threshold, or carries large traceability/appendix tables. The
extract must preserve: product thesis, MVP-0/MVP-1, actors, capabilities,
workflow model, logical architecture, conceptual information model, decision
policies, risks and release gates, evaluation requirements, technical-design
handoff notes, the design decision register (if present), and open questions.
Small blueprints pass through unchanged — that is why the task is named
`prepare_blueprint_context`, not `compact_blueprint`. Never silently discard
security, observability, AI-boundary, or MVP information.
