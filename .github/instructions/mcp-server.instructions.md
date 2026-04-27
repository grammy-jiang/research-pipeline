---
applyTo: "src/research_pipeline/mcp_server/**/*.py"
---

- Tool implementations in `tools.py` are thin adapters — business logic stays in `src/research_pipeline/`.
- Every tool has a matching Pydantic schema in `schemas.py`.
- When adding a new tool: add schema → implement in tools.py → register in server.py → add tests.
- Keep the MCP server runnable via `research-pipeline mcp serve`.
