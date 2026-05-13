# Daily AI Intelligence — Final Completeness Audit Pack

This is not a new product phase.

It is a final independent verification stage to prove that all A-G features/functions from the plan are implemented, tested, documented, and reachable through the intended CLI/MCP/skill surfaces.

## Purpose

The audit answers four questions:

1. Does every planned feature exist?
2. Is every feature reachable through the expected user/agent-facing surface?
3. Is every feature verified by deterministic tests, validators, proof packs, and acceptance gates?
4. Did any implementation drift from the original non-goals or governance rules?

## Copy into repository root

Copy `.github/`, `docs/`, and `prompts/` into the repository root.

## Existing `copilot-instructions.md`

Do not overwrite repository instructions.

Append or replace the current Daily AI Intelligence block with:

```text
.github/copilot-instructions.daily-ai-intelligence-final-audit.append-or-replace.md
```

## Recommended Copilot command

```bash
copilot --autopilot --allow-all --max-autopilot-continues 20 \
  -p "$(cat prompts/daily-ai-intelligence/final-completeness-audit.md)"
```

## Important rule

The final audit must not silently implement new features.

If a gap is found, reopen the original phase ticket or create a narrowly scoped gap-fix ticket mapped back to the original phase.
