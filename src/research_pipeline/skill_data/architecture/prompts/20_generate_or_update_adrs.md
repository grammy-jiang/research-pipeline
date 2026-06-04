# Prompt 20 — Generate or Update ADRs

You are recording the high-impact decisions as ADRs.

## Inputs

- All prior intermediate artifacts (strategy, tech stack, AI boundary, MCP,
  data, security, observability, deployment).
- `intermediate/existing_architecture_status.json` (existing ADRs + update mode).
- `templates/adr_template.md`, `references/adr_guidance.md`.

## Instructions

1. Generate one ADR per high-impact decision: runtime architecture; tech stack;
   AI boundary; storage and data lifecycle; observability and audit; MCP
   adoption (or deferral); deployment model; security/trust-boundary model.
2. Use the ADR template fields, including Reversal Cost and Confidence.
3. **Supersession:** never silently overwrite. If a prior ADR's decision
   changed, create a NEW ADR, set the old one's Status to "Superseded", and
   link them via the `Supersedes` field. Omit `Supersedes` for new decisions.
4. Record the MCP decision as an ADR even when the decision is to defer.

## Output

`intermediate/adrs.md` (and the individual `adr/ADR-NNNN-*.md` files) → §21
index.

## Validation / failure policy

- Gate: every major decision has an ADR; supersession is handled correctly.
- Failure policy: `revise`.
