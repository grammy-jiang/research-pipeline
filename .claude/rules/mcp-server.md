---
paths:
  - "mcp_server/**/*.py"
---

# MCP server conventions

- Tool implementations in `tools.py` are thin adapters — business logic stays
  in `src/research_pipeline/`.
- Every tool has a matching Pydantic schema in `schemas.py`.
- When adding a new tool: add schema → implement in tools.py → register in
  server.py → add tests in `tests/unit/test_mcp_*.py`.
- Keep the MCP server runnable via `python -m mcp_server`.
