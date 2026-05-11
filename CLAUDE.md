@AGENTS.md

<!-- HC1-HC6 hard constraints defined in AGENTS.md govern all agent sessions.
     Overlays may specialize but must never relax those constraints. -->

## Claude Code

- Use plan mode for changes touching `src/research_pipeline/pipeline/` or
  `src/research_pipeline/config/` (high-impact modules).
- When adding new CLI commands, check existing `cmd_*.py` files for the
  established pattern before writing code.
- For MCP server changes, always update both `src/research_pipeline/mcp_server/tools.py` and
  `src/research_pipeline/mcp_server/schemas.py` together.
