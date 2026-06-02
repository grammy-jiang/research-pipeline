# Research Report: Notes on AI Agent Memory

> Illustrative *weak* excerpt. It has a research question and a couple of
> loosely evidenced mechanisms, but no confidence grades, no gap
> classification, and a thin reference list. The `blueprint` skill should
> classify this as `weak` (not `insufficient`): proceed, but mark missing
> areas as assumptions / open questions.

## Research Question

What would it take to give AI coding agents useful memory across sessions?

## Notes

- Some systems store past interactions and retrieve them later; this seems
  to help, though the evidence here is anecdotal [2312.01234].
- Keyword search and embedding search both get used; unclear which is
  better in practice.
- People worry about agents remembering wrong or malicious things, but no
  rigorous treatment is collected here.

## References

- [2312.01234] A memory system for agents.

---

> For contrast, an `insufficient` input would have *no* research question
> and *no* mechanisms or evidence at all — only a topic title. In that case
> the skill must STOP and emit the standardized insufficient-input failure
> document (see `references/troubleshooting.md`), not a blueprint.
