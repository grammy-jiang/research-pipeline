# Prompt 03 — Prepare Blueprint Context

You are preparing the blueprint for reliable architecture reasoning. Compaction
is **conditional** — small blueprints pass through unchanged.

## Inputs

- `intermediate/input_resolution.json` (blueprint path).
- `intermediate/existing_architecture_status.json` (update mode).
- The blueprint file itself.

## Instructions

1. If the blueprint is small enough for reliable full-context reasoning, pass
   it through unchanged and say so.
2. If it is large (exceeds the practical context budget, or carries large
   traceability/appendix tables), create an architecture-focused extract
   following `references/input-discovery.md` "Context budget".
3. The extract must preserve: product thesis; MVP-0/MVP-1; actors;
   capabilities; workflow model; logical architecture; conceptual information
   model; decision policies; risks and release gates; evaluation requirements;
   technical-design handoff notes; design decision register (if present); open
   questions. Compress long evidence tables; keep references to the original
   blueprint sections.
4. In `patch`/`adr-only`/`compare` modes, also keep the parts of the blueprint
   that map to the changed sections.
5. Never silently discard security, observability, AI-boundary, or MVP
   information.

## Output

`intermediate/blueprint_architecture_extract.md` (or a note that the full
blueprint is used unchanged, plus a one-line reason).

## Validation / failure policy

- Gate: context prepared without losing MVP, risks, interfaces, or handoff.
- Failure policy: `revise`.
