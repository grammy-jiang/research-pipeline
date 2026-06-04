# Reference: MCP Adoption Guide

Load when deciding skill vs MCP boundaries (prompt 11). MCP must be justified,
never adopted because it is fashionable.

## Use a Skill when

```text
the task is a repeatable reasoning workflow
the task benefits from structured prompts
the task is invoked by a human or another agent
the task does not require stable external system access
```

## Use an MCP Server when

```text
multiple AI clients need reusable access to a capability
agents need permissioned access to data/tools
the capability should expose resources/tools/prompts
the boundary must be reusable outside this one skill
```

## MCP adoption gate — FAIL architecture review if MCP is introduced without

```text
clear external clients
clear resources/tools exposed
a permission boundary
audit requirements
an error model
a versioning approach
a non-MCP alternative considered
```

## Decision procedure

1. State the capability and who needs it.
2. If only this skill/agent needs it and no stable external access is required
   → **Skill** (or an internal module). Stop.
3. If multiple clients need permissioned, reusable access → consider **MCP**.
4. Run the adoption gate above. If any item is missing, either supply it or
   **DEFER** MCP and record the deferral as an ADR.
5. Record the outcome in §11.2 and as ADR-00xx (mcp-adoption), even when the
   decision is to defer.

## Default posture

> Prefer a skill or an internal module first. Adopt MCP only when reusable,
> permissioned, multi-client external access is a real, present requirement —
> not a possible future one. A deferred MCP decision is a valid, documented
> outcome.
