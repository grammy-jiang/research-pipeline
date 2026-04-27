---
name: daily-ai-intelligence
description: Run private daily AI technical intelligence briefings with governed sources, ranked evidence packs, validation, feedback, Obsidian export, and optional dossiers.
license: MIT
compatibility: GitHub Copilot CLI, Claude Code, MCP clients
---

# Daily AI Intelligence

Use this skill when the user asks for a daily AI technical brief, AI tooling
watch, coding-agent update scan, MCP/Copilot/Claude Code daily brief, weekly AI
tooling trend memo, or a hot-topic dossier from a daily brief.

Do not use this skill for academic literature reviews, PDF download/conversion,
citation expansion, or paper-only synthesis. Hand those requests to the
`research-pipeline` academic workflow.

## Core workflow

1. Poll only configured sources: `research-pipeline brief poll --registry config.toml`.
2. Rank deterministically: `research-pipeline brief rank`.
3. Generate and validate: `research-pipeline brief generate-daily` then
   `research-pipeline brief validate`.
4. For one-step runs use `research-pipeline brief run --registry config.toml`.
5. Record explicit feedback after review with `research-pipeline brief feedback`.
6. Export to Obsidian only when a vault path is configured.

Never send raw unclustered source dumps to a cloud model. Use ranked evidence
packs and validated reports only.

See `references/command-reference.md`, `references/source-policy.md`, and
`references/agent-evaluation.md`.
