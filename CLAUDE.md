@AGENTS.md

## Claude Code

- Use plan mode for changes touching `src/research_pipeline/pipeline/` or
  `src/research_pipeline/config/` (high-impact modules).
- When adding new CLI commands, check existing `cmd_*.py` files for the
  established pattern before writing code.
- For MCP server changes, always update both `mcp_server/tools.py` and
  `mcp_server/schemas.py` together.
