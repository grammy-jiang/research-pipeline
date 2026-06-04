# Update History Template

Place this block near the top of the document, immediately after §1 (and its
generation metadata). Every architecture document must include it.

```markdown
## Update History

| Date | Source Blueprint | Architecture Version | Change Type | Affected Sections | Notes |
|---|---|---|---|---|---|
| <YYYY-MM-DD> | `<blueprint filename>` | <version> | <change type> | <sections> | <notes> |
```

Rules:

- New architecture documents include an **initial** row (Change Type `initial`,
  Affected Sections `all`).
- Updated documents **append** a new row.
- Never delete previous update-history rows.
- Change Type ∈ {`initial`, `regenerate`, `patch`, `adr-only`, `resume`,
  `compare`}.
- For `patch`/`adr-only`, list the specific affected sections (e.g.
  "§7, §9, §20, §21").
