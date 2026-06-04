# Generation Metadata Template

Place this block as §1.x inside §1 Executive Architecture Summary.

```markdown
### 1.x Generation Metadata

| Field | Value |
|---|---|
| Source blueprint | `<filename>` |
| Source blueprint version/hash | `<hash or timestamp or unknown>` |
| Source blueprint generated at | `<date or unknown>` |
| Architecture skill version | `<from manifest.json version or unknown>` |
| Generated at | `<date>` |
| Operating mode | interactive / automatic / hybrid |
| Clarification count | `<N>` |
| Assumptions made | `<N>` |
| Output detail | concise / standard / detailed |
| Target deployment assumption | local / server / cloud / hybrid / unknown |
```

Rules:

- Do not invent metadata. Use `unknown` when a value is unavailable.
- Record assumptions separately from known facts.
- The architecture skill version comes from `manifest.json` (`version`). If you
  cannot read it, write `unknown` — do not invent a number.
- The source-blueprint hash/timestamp comes from the file; if not derivable,
  write `unknown`.
