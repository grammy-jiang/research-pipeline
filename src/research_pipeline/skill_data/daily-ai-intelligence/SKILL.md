---
name: daily-ai-intelligence
description: >
  Run the daily AI technical-intelligence briefing workflow. Polls governed
  registry sources (GitHub releases, RSS/Atom, Hacker News, Reddit, Bluesky,
  X, papers, video/audio, manual items), normalizes and deduplicates events,
  ranks with topic memory and explicit feedback, generates a validated daily
  Markdown brief, and optionally exports to Obsidian, builds hot-topic
  dossiers, records feedback, runs weekly trend synthesis, or compares ranked
  output across registries. Use when the user asks for a "daily AI brief",
  "AI tooling watch", "coding-agent update scan",
  "MCP/Copilot/Claude Code daily brief", "weekly AI tooling trend memo",
  "hot-topic dossier from a daily brief", a "weekly AI tooling synthesis",
  or to "rank/validate/export today's AI intelligence". Do NOT use for
  academic literature reviews, PDF download/conversion, citation-graph
  expansion, or paper-only synthesis (use the `research-pipeline` skill
  instead).
license: MIT
---

# Daily AI Intelligence

## When To Trigger

- "Run today's AI brief" / "daily AI technical brief" / "AI tooling watch"
- "Coding-agent update scan" / "MCP daily brief" / "Copilot/Claude Code digest"
- "Weekly AI tooling trend memo" / "Weekly synthesis from my daily briefs"
- "Generate a hot-topic dossier for cluster X"
- "Record feedback on this cluster/source/topic"
- "Export today's brief to my Obsidian vault"

Do **not** trigger for academic literature reviews, paper-only synthesis,
PDF conversion, general web search, or social-media scraping outside the
reviewed registry. Do not use this skill for academic research — hand off
to `research-pipeline` (academic) as needed.

## Launch

**Always launch through the manifest-governed runner. Never bypass it.**

```bash
SKILL_DIR=~/.claude/skills/daily-ai-intelligence     # Claude Code
# SKILL_DIR=~/.copilot/skills/daily-ai-intelligence  # Copilot CLI
# SKILL_DIR=~/.agents/skills/daily-ai-intelligence   # Codex CLI
REG=~/my-daily-registry.toml   # user's reviewed registry
WS=./workspace/briefing

python3 $SKILL_DIR/runners/runner.py --registry "$REG" --workspace "$WS"
```

The runner reads `manifest.json`, initialises `workflow_state.json` under
`<WS>/<date>/`, and drives all tasks in dependency order via the
`research-pipeline brief` CLI/MCP tools. Optional tasks
(review-ranked, dossier, feedback, export-obsidian, weekly-synthesis) are
triggered only when the corresponding context key is set (via `--reviewer`,
`--cluster-id`, `--vault`, `--week`). Completion is proved by artifact
existence + schema validation — not by agent claim.

## Rules

1. **Do not bypass the runner.** Never call `brief` CLI commands directly
   without the orchestrator updating `workflow_state.json`.
2. **Resume = re-run the runner.** Pass `--state <existing>.json` to
   continue an interrupted workflow. Idempotent: accepted tasks are skipped.
3. **Reviewer gates.** If the optional rank_reviewer returns `status: "rejected"`,
   fix ranked events, reset the task to `pending`, and re-run. Do not
   override a `rejected` verdict.
4. **Validate before delivery.** Never surface a brief whose
   `validation/validation.json` reports failures. Low-signal days are
   valid outputs — do not pad them.
5. **Registry-only sources.** Never add ad-hoc scraping. New sources
   require an explicit reviewed registry entry.
6. **Privacy gate.** Never send raw, unclustered source dumps to a
   cloud LLM. Only ranked evidence packs and validated reports may
   leave the local workspace.

## References

| File | Load when |
|------|-----------|
| `references/workflow-steps.md` | Per-agent SKILL_DIR paths, or understanding/diagnosing an orchestrated task |
| `references/command-reference.md` | All `brief` subcommands, options, MCP tool map |
| `references/source-policy.md` | Adding/enabling a source or refusing expansion |
| `references/feedback-loop.md` | Recording feedback or computing preference adjustments |
| `references/report-templates.md` | Writing or validating a daily/weekly report |
| `references/agent-evaluation.md` | Evaluating or demonstrating the agent |
| `references/troubleshooting.md` | Empty reports, validation failures, dedup/vault errors |

## Final Response To User

When `workflow_state.json` shows `status: complete`:

1. Show the brief path and the validation result summary.
2. List the top 5 clusters and their event counts.
3. Offer to build a dossier, export to Obsidian, or run weekly synthesis.
