---
name: daily-ai-intelligence
description: >
  Run a private daily AI technical-intelligence briefing using the
  research-pipeline `brief` CLI/MCP workflow. Polls only governed
  registry sources (GitHub releases, RSS/Atom, Hacker News, Reddit,
  Bluesky, X, papers, video/audio, academic enrichment, manual
  curated items), normalizes and deduplicates events, ranks
  deterministically with topic memory and explicit feedback,
  generates a validated daily Markdown brief, and optionally
  exports to Obsidian, builds hot-topic dossiers, records
  feedback, applies reversible preference adjustments, runs
  weekly trend synthesis, or compares ranked output across
  registries. Use when the user asks for a "daily AI brief",
  "AI tooling watch", "coding-agent update scan",
  "MCP/Copilot/Claude Code daily brief", "weekly AI tooling
  trend memo", "hot-topic dossier from a daily brief", a
  "weekly AI tooling synthesis", or to "rank/validate/export
  today's AI intelligence". Do NOT use for academic literature
  reviews, PDF download/conversion, citation-graph expansion,
  or paper-only synthesis (hand off to the `research-pipeline`
  academic skill).
license: MIT
compatibility: >
  Requires the research-pipeline CLI (`pip install research-pipeline`)
  and a reviewed source registry (TOML/JSON/YAML). Network access is
  needed only for enabled live sources; fixtures-only runs work
  fully offline. Works in Claude Code, Claude.ai, GitHub Copilot
  CLI, Codex, and any MCP client that can invoke the `tool_brief_*`
  family.
metadata:
  author: grammy-jiang
  version: 1.0.0
  category: intelligence
  tags: [daily-brief, ai-tooling, mcp, copilot, claude-code, governance, feedback, obsidian, dossier, weekly-synthesis]
---

# Daily AI Intelligence

Use this skill to run the `research-pipeline brief` CLI/MCP workflow and
produce a deterministic, evidence-cited daily AI technical-intelligence
brief from a governed source registry. Keep `SKILL.md` light; load
reference files only when that detail is needed.

## When To Trigger

Trigger on requests like:

- "Run today's AI brief" / "daily AI technical brief" /
  "Daily AI intelligence" / "AI tooling watch"
- "Coding-agent update scan" / "MCP daily brief" /
  "Copilot/Claude Code update digest"
- "Weekly AI tooling trend memo" / "Weekly synthesis from my daily briefs"
- "Generate a hot-topic dossier for cluster X" /
  "Dossier from yesterday's brief"
- "Record feedback on this cluster/source/topic"
- "Export today's brief to my Obsidian vault"
- "Compare ranked output with and without source <S>"

Do **not** trigger for:

Do not use this skill for the cases below — hand off as noted:

- Academic literature reviews, paper-only synthesis, or citation-graph
  expansion — hand off to the `research-pipeline` skill.
- Single PDF-to-Markdown conversion — use
  `research-pipeline convert-file <pdf>` or the `pdf-to-markdown` skill.
- General web search, news lookups, or social-media scraping outside the
  reviewed registry — refuse and ask the user to add a registry entry.
- Requirements clarification or architecture design — hand off to the
  `req-analysis` skill.

## Critical Rules

1. **Use the installed CLI directly.** `research-pipeline brief ...` —
   not `uv run` or `python -m`, unless developing this repository.
2. **Always pass `--registry REG`**, where `REG` is the user's reviewed
   registry. The bundled template lives at:
   - `~/.claude/skills/daily-ai-intelligence/config.toml` (Claude Code / Copilot)
   - `~/.codex/skills/daily-ai-intelligence/config.toml` (Codex)
   Copy it once and edit, do not poll the disabled example source.
3. **Always pass `--workspace WS`.** Default to `./workspace/briefing`.
   Every run writes to `<WS>/<YYYY-MM-DD>/...`. Do not mix workspaces
   between users or topics — the global topic-memory and feedback stores
   live inside `WS`.
4. **Registry-only sources.** Never add ad-hoc scraping, social
   firehoses, or general repository surveillance. New sources require an
   explicit registry entry (cadence, retention policy, fixtures, trust
   weight, review). See `references/source-policy.md`.
5. **Privacy & cost gate.** Never send raw, unclustered source dumps
   to a cloud LLM. Only ranked evidence packs and validated reports
   may leave the local workspace. Manual items follow the same
   normalization, dedup, ranking, validation, and budget rules.
6. **Determinism.** Ranking is deterministic given inputs + feedback +
   topic memory. Do not swap in random tie-breakers, sampling
   re-rankers, or model-graded scores inside the ranking stage.
   Cloud-model post-processing happens *after* `validate`.
7. **Validate, then deliver.** Always run
   `research-pipeline brief validate` and refuse to surface a brief
   whose `validation/validation.json` reports failures. Low-signal /
   no-news days are valid outputs — do not pad them.
8. **Feedback first, behavior second.** Use explicit feedback signals
   (`keep`, `hide`, `more_like_this`, `less_like_this`, `too_noisy`,
   `already_known`, `not_actionable`, `useful`, `neutral`,
   `not_useful`, `wrong_cadence`). Never promote behavioral signals
   into durable ranking without curator review. All preference
   adjustments are reversible via
   `research-pipeline brief preferences --rollback`.
9. **Obsidian export is opt-in and allowlisted.** Only export when the
   user supplies `--vault` and the target subdir is in the registry
   allowlist. Never overwrite a note that lacks the
   generator-owned frontmatter — surface the conflict and stop.
10. **Hot-topic dossiers are scoped.** Build at most a small number per
    day (default `--max-count 1` in `--auto`) and only from clusters
    that meet the dossier threshold. Do not synthesize dossiers from
    raw event lists.

## Load References

Load each reference only when that level of detail is needed:

| File | When to load |
|------|--------------|
| `references/command-reference.md` | All `brief` subcommands, options, MCP tool map. |
| `references/source-policy.md` | Before adding/enabling a source or refusing an expansion request. |
| `references/feedback-loop.md` | Before recording feedback or computing preference adjustments. |
| `references/report-templates.md` | Before writing or validating a daily/weekly report. |
| `references/agent-evaluation.md` | When asked to demonstrate or evaluate the agent on held-out tasks. |
| `references/troubleshooting.md` | Empty reports, validation failures, dedup or vault errors. |

## Default Workflow

Set:

```bash
# Claude Code / GitHub Copilot
REG=~/.claude/skills/daily-ai-intelligence/config.toml
# Codex
# REG=~/.codex/skills/daily-ai-intelligence/config.toml

WS=./workspace/briefing                      # workspace root
DATE=$(date -u +%F)                          # today, UTC
```

Make sure `REG` is the user's reviewed registry, not the bundled
disabled example. Copy and edit it on first use:

```bash
cp ~/.claude/skills/daily-ai-intelligence/config.toml ~/my-daily-registry.toml
# edit ~/my-daily-registry.toml: enable real sources, set cadences, fixtures
REG=~/my-daily-registry.toml
```

### 1. One-shot daily run (preferred)

```bash
research-pipeline brief run --registry "$REG" --workspace "$WS" --date "$DATE"
```

This runs `poll → rank → generate-daily → validate` in order. Output:

- `<WS>/<DATE>/raw/*.jsonl` — raw events per source
- `<WS>/<DATE>/normalized/events.jsonl` — deduped, normalized
- `<WS>/<DATE>/clusters/clusters.jsonl` — clusters + ranked clusters
- `<WS>/<DATE>/reports/daily.md` — the validated daily brief
- `<WS>/<DATE>/validation/validation.json` — pass/fail + reasons

### 2. Step-by-step (debugging, partial reruns)

```bash
research-pipeline brief poll           --registry "$REG" --workspace "$WS" --date "$DATE"
research-pipeline brief rank           --workspace "$WS" --date "$DATE"
research-pipeline brief generate-daily --workspace "$WS" --date "$DATE"
research-pipeline brief validate       --workspace "$WS" --date "$DATE"
```

Use `research-pipeline brief resume --workspace "$WS" --date "$DATE"`
to skip stages whose artifacts already exist on disk.

### 3. Drill into a hot topic (dossier)

```bash
research-pipeline brief dossier --workspace "$WS" --date "$DATE" --cluster <CLUSTER_ID>
# or auto-select up to N (default 1):
research-pipeline brief dossier --workspace "$WS" --date "$DATE" --auto --max-count 1
```

### 4. Record feedback and apply reversible preference updates

```bash
research-pipeline brief feedback --workspace "$WS" --date "$DATE" \
    --cluster <CLUSTER_ID> --signal keep --reason "tracking this MCP RFC"
research-pipeline brief feedback --workspace "$WS" --date "$DATE" \
    --topic   <TOPIC_ID>   --signal too_noisy
research-pipeline brief feedback --workspace "$WS" --date "$DATE" \
    --source  <SOURCE_ID>  --signal less_like_this

research-pipeline brief preferences --workspace "$WS"            # apply
research-pipeline brief preferences --workspace "$WS" --rollback # undo
```

### 5. Export to Obsidian (opt-in, allowlisted)

```bash
research-pipeline brief export-obsidian \
    --vault    /path/to/Obsidian/AI-Intelligence \
    --registry "$REG" \
    --workspace "$WS" \
    --date "$DATE"
```

Use `--dry-run` first to preview target paths. The export refuses to
overwrite notes that lack the generator-owned frontmatter.

### 6. Weekly synthesis & source comparison

```bash
research-pipeline brief weekly-synthesis --workspace "$WS" --week "$(date -u +%G-W%V)"
research-pipeline brief compare-sources  --registry "$REG" --workspace "$WS" --date "$DATE" \
    --expanded-registry ~/proposed-registry.toml
```

`compare-sources` answers "would adding/removing source S change
today's ranked output?" without touching the production registry.

### 7. Topic alias review

```bash
research-pipeline brief topic-aliases --workspace "$WS" --list
research-pipeline brief topic-aliases --workspace "$WS" --approve <ALIAS_ID>
research-pipeline brief topic-aliases --workspace "$WS" --reject  <ALIAS_ID>
```

Durable topic aliases require explicit approval — never auto-approve.

## Schedule

Cron (06:00 local, with logs):

```cron
0 6 * * * cd ~/projects && research-pipeline brief run \
    --registry ~/my-daily-registry.toml \
    --workspace ~/workspace/briefing \
    >> ~/.local/state/daily-brief.log 2>&1
```

`--date` defaults to today (UTC); omit it for cron.

## MCP Usage

The MCP server exposes the same workflow under the `tool_brief_*` family:

- `tool_brief_run`, `tool_brief_poll`, `tool_brief_rank`,
  `tool_brief_generate_daily`, `tool_brief_validate`
- `tool_brief_feedback`, `tool_brief_preferences`,
  `tool_brief_topic_aliases`
- `tool_brief_export_obsidian`, `tool_brief_dossier`,
  `tool_brief_weekly_synthesis`, `tool_brief_compare_sources`,
  `tool_brief_resume`

Resources expose the daily brief, validation report, ranked clusters,
feedback ledger, and topic memory at URIs like
`briefing://<workspace>/<date>/daily.md`. Prefer the resource over
re-reading the file. See `references/command-reference.md` for the
current tool/resource map.

## Final Response To User

After completing a run, report:

- final brief path (`<WS>/<DATE>/reports/daily.md`);
- counts: sources polled, events normalized, clusters ranked, items
  surfaced, items suppressed, watchlist-quiet flag;
- validation result (pass/fail + reasons if failed);
- top items with cluster IDs, source IDs, and confidence;
- any dossier(s) generated and their cluster IDs;
- explicit feedback recorded and whether
  `brief preferences` was applied;
- Obsidian export status (vault path, files written, conflicts);
- next-step suggestions: "record feedback on cluster X",
  "promote topic Y", "review topic-alias suggestion Z",
  "schedule weekly synthesis on Sunday".

## Examples

Concrete user request → skill behavior. Use these to calibrate
triggering and default actions.

### Example 1 — Fresh daily run

User: *"Run today's AI tooling brief."*

Actions:

1. Resolve `REG` from `~/.claude/skills/daily-ai-intelligence/config.toml`
   (or the user's edited copy).
2. `research-pipeline brief run --registry "$REG" --workspace ./workspace/briefing`.
3. Read `validation/validation.json`; refuse to deliver on failure.
4. Surface `reports/daily.md` and the executive signal, top items,
   suppressed items, and watchlist-quiet flag.

### Example 2 — Hot-topic dossier from yesterday's brief

User: *"Build a dossier on the new MCP elicitation RFC from yesterday."*

Actions:

1. Locate the matching cluster ID in
   `<WS>/<YESTERDAY>/clusters/ranked.jsonl`.
2. `research-pipeline brief dossier --workspace "$WS" --date "$YESTERDAY" --cluster <ID>`.
3. Surface the dossier path, its primary-artifact citations, and any
   open questions for follow-up feedback.

### Example 3 — Feedback + reversible preference update

User: *"Mark the 'Copilot CLI' cluster as keep, the 'crypto' topic as too noisy, and apply preferences."*

Actions:

1. `brief feedback --cluster <ID> --signal keep`.
2. `brief feedback --topic crypto --signal too_noisy`.
3. `brief preferences --workspace "$WS"` to apply the deltas.
4. Tell the user how to roll back with
   `brief preferences --rollback`.

### Example 4 — Weekly trend memo

User: *"Give me a weekly AI tooling synthesis for last week."*

Actions:

1. `research-pipeline brief weekly-synthesis --workspace "$WS" --week 2026-W17`.
2. Surface the memo path and its top recurring themes, contradictions,
   and watchlist-quiet sources for the week.

### Example 5 — Refuse unsupported source expansion

User: *"Also scrape r/MachineLearning and Twitter for me."*

Actions:

1. Refuse — these are not in the registry.
2. Quote `references/source-policy.md` and offer to draft a registry
   entry (cadence, retention, fixture, trust/noise weights) for the
   user to review before enabling.

### Example 6 — Out of scope, redirect

User: *"Do a literature review on transformer time-series papers."*

Actions:

1. Do **not** invoke `brief`.
2. Hand off to the `research-pipeline` academic skill.
