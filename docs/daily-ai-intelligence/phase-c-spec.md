# Daily AI Intelligence — Phase C Spec

## Purpose

Export daily intelligence outputs to Obsidian so they are searchable, reviewable, and agent-readable.

## Preconditions

A01-A12 and B01-B08 are `audit_pass`; phases A/B are complete; phase C is in_progress.

## Scope

Vault path allowlist, path safety checks, daily/topic/source notes, YAML frontmatter, stable local note paths, wiki-links/backlinks, idempotent generated-note updates, validators, `brief export-obsidian`, dry-run support, offline e2e tests.

## Non-goals

No dossiers, feedback learning, behavioral tracking, social sources, MCP expansion, or UI.

## Vault layout

```text
AI-Intelligence/
  Daily/YYYY-MM-DD.md
  Topics/<topic-slug>.md
  Sources/<source-id>.md
  Weekly/
  Monthly/
```

Dossiers are reserved for Phase E.

## Ownership rule

A note can be updated automatically only if it has matching generated-note frontmatter ID. Existing human notes without that ID must not be overwritten.

## Wiki-link rule

Preserve wiki-links exactly. Do not convert wiki-links to Markdown links.

## Dry-run rule

Write commands must support dry-run and report planned changes without writing.

## Completion gate

C01-C08 audit_pass; daily/topic/source notes export; unsafe paths rejected; human notes not overwritten; wiki-links preserved; exports idempotent; Phase A/B tests still pass.
