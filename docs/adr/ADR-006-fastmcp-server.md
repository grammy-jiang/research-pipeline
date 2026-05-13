# ADR-006: FastMCP for the MCP Server

## Status
Accepted

## Date
2024

## Context

The Model Context Protocol (MCP) enables AI assistants (Claude, GitHub Copilot,
etc.) to call tools and access resources in the pipeline. Two implementation
options were considered:

1. **Raw `mcp` Python SDK** — maximum control, verbose boilerplate for every tool
2. **FastMCP** — high-level wrapper that uses function signatures and docstrings
   to auto-generate tool schemas

The pipeline exposes 64 tools, 21 resources, and 6 prompts. Without a high-level
abstraction, writing schema boilerplate for 64 tools would be ~3000 lines of
repetitive JSON schema definitions.

## Decision

Use **FastMCP** as the MCP server framework. Tools are defined as Python functions
with type-hinted parameters and docstrings — FastMCP generates the JSON schema
automatically.

All tools are registered in `src/research_pipeline/mcp_server/tools.py`. Input
schemas are defined as Pydantic models in `src/research_pipeline/mcp_server/schemas.py`.
MCP tools are thin adapters over the same CLI logic to keep the two surfaces in sync.

Features used:
- Tool annotations (read-only vs. mutating)
- Progress reporting for long-running tools
- Resource handlers with URI templates
- Prompt templates
- Auto-completions for `run_id` parameters

## Consequences

**Positive:**
- 64 tools defined with minimal boilerplate
- Type safety preserved (Pydantic schemas validated before tool execution)
- Tool schema automatically updates when function signature changes
- Progress reporting lets AI assistants display real-time feedback

**Negative:**
- FastMCP is a third-party dependency; its API could break between releases
- Abstraction hides some low-level MCP protocol control
- Tools must be thin adapters — business logic must not live in MCP layer

**Versioning:** The MCP server version tracks the package version. Any breaking
change to tool signatures requires a `feat:` commit and changelog entry.
