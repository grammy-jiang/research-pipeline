# Security Model: research-pipeline

## 1. Document Purpose

This document describes the complete security model for `research-pipeline`,
covering governance constraints (HC1–HC6), boundary controls, MCP tool security,
adversarial robustness, content taint tracking, and the network egress policy.

---

## 2. Executive Summary

`research-pipeline` processes untrusted content from external academic sources
(paper abstracts, PDFs, API responses) and exposes its capabilities via a CLI
and an MCP server. The security model is designed around:

1. **Hard constraints (HC1–HC6)** — immutable governance rules enforced at the
   agent and CI levels.
2. **MCP tool security** — zero-trust tool registry with hash-pinned schemas,
   trust domains, and audit trails.
3. **Content boundary gates** — taint tracking for external content as it flows
   through pipeline stages.
4. **Adversarial robustness** — ToolTweak-style perturbation catalog and
   defense-trilemma K^n budget monitor.
5. **Secret management** — `detect-secrets` scanning; all credentials in
   `config.toml` / environment variables only.

---

## 3. Hard Constraints (HC1–HC6)

These constraints are **immutable governance rules**. They apply to every
agent session and every contributor. No runtime overlay or agent instruction
may relax them.

| ID | Rule | Enforcement |
|----|------|-------------|
| **HC1** | No plaintext secrets in repository files, prompts, logs, or commits. | `detect-secrets` pre-commit hook; `.secrets.baseline` in repo |
| **HC2** | No agent-authored writes outside the path allowlist. | AGENTS.md, agent instructions |
| **HC3** | Destructive commands (`rm -rf`, `git push --force`, `DROP TABLE`, etc.) require explicit human approval before execution. | Convention + agent instructions |
| **HC4** | Database schema changes (migrations, drops) must be authored but never executed autonomously. | Convention + agent instructions |
| **HC5** | Network egress from agent-executed code is limited to the approved destination list. | Config + convention |
| **HC6** | Red-class data (secrets, PII, credentials, API keys) must never enter prompts, tool arguments, trace logs, or stored artefacts. | Pre-commit secret scanning; config pattern |

### HC2 — Write allowlist

Agent-authored writes are permitted only in these paths:

```
src/
tests/
docs/
pyproject.toml
.pre-commit-config.yaml
Makefile
AGENTS.md
CLAUDE.md
.github/
```

Any write outside this list must be denied and reverted.

### HC5 — Approved network destinations

```
arxiv.org
export.arxiv.org
api.semanticscholar.org
api.openalex.org
dblp.org
serpapi.com
pypi.org
files.pythonhosted.org
github.com
```

All other destinations require explicit human approval before code is executed.

---

## 4. Secret Management

### 4.1 Pre-commit secret scanning

Every commit is scanned by `detect-secrets` (v1.5.0) using the pre-commit hook:

```yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.5.0
  hooks:
    - id: detect-secrets
      args: ['--baseline', '.secrets.baseline']
      exclude: '(uv\.lock|\.secrets\.baseline)'
```

The `.secrets.baseline` file records known false-positives. New secrets trigger
a CI failure.

### 4.2 Configuration of credentials

API keys and other credentials must be stored in **one of two locations only**:

1. `config.toml` — gitignored; never committed
2. Environment variables — e.g., `SERPAPI_KEY`, `MATHPIX_APP_ID`

The `config.example.toml` shows all configuration slots with placeholder values.
The `RESEARCH_PIPELINE_CONFIG` environment variable points to the config file path.

### 4.3 Log sanitization

The pipeline never logs API keys or credentials. The `infra/sanitize.py` module
provides `sanitize_text()` for stripping sensitive patterns from content before
it enters logs or stored artefacts.

---

## 5. MCP Tool Security

The MCP server implements a 4-layer zero-trust defense (MCPSHIELD-inspired)
via `src/research_pipeline/security/mcp_guard.py`.

### 5.1 Tool registry and hash-pinning

```python
from research_pipeline.security.mcp_guard import ToolRegistry, McpGuard

registry = ToolRegistry()
registry.register("search", schema={"query": "str"}, domain="read")
registry.pin_tool("search")   # Computes SHA-256 hash of schema

guard = McpGuard(registry)
result = guard.authorize("search", {"query": "transformers"}, caller="pipeline")
if result.allowed:
    execute_tool(result.tool_name, result.args)
```

Hash-pinning detects **ToolTweak-style attacks**: small perturbations to a tool
name or description that could alter agent selection behaviour.

### 5.2 Trust domains

| Domain | Permitted operations |
|--------|---------------------|
| `read` | Query APIs, read files, search |
| `write` | Create/update workspace files |
| `execute` | Run sub-processes (PDF conversion backends) |
| `network` | HTTP calls to external services |
| `system` | File system operations, index management |

### 5.3 Authorization decisions

| Decision | Meaning |
|----------|---------|
| `allowed` | Tool invocation permitted |
| `denied` | Tool invocation blocked; error returned to caller |
| `requires_approval` | Destructive operation; human confirmation needed |

### 5.4 Audit trail

Every tool invocation is logged with:
- timestamp
- tool name + schema hash
- calling context (caller identifier)
- authorization decision
- arguments (with sensitive values redacted)

---

## 6. Content Taint Tracking

External content (paper abstracts, PDF text, API responses) is classified and
tracked as it flows through pipeline stages.

### 6.1 Taint labels

```python
class TrustLevel(StrEnum):
    TRUSTED = "trusted"          # First-party or verified content
    SEMI_TRUSTED = "semi_trusted"# Published academic content
    UNTRUSTED = "untrusted"      # Raw web scrapes, user-supplied
```

A `TaintLabel` records: `source`, `stage`, `trust_level`, `sanitized`,
`classified`, `risk_flags`.

### 6.2 Taint propagation

If tainted content is used to generate new content, the new content inherits
the taint. This prevents prompt-injection vectors from being laundered through
the pipeline.

### 6.3 Security gates

Gates run at each stage boundary where external content enters:

| Stage | Gate trigger |
|-------|-------------|
| `search` | Abstracts and titles from search APIs |
| `download` | PDF files from arXiv/external URLs |
| `convert` | Extracted Markdown text from PDFs |
| `extract` | Chunked content entering the indexer |

Each gate performs:
1. **Classify** — determine trust level
2. **Sanitize** — strip unsafe patterns if needed
3. **Record taint** — attach label to content key

Implementation: `src/research_pipeline/security/gates.py`

---

## 7. Adversarial Robustness

### 7.1 ToolTweak perturbation catalog

`src/research_pipeline/security/adversarial.py` implements 10 adversarial
perturbations for MCP tool definitions, used in tests to verify the hash-pinning
system catches tampering:

| Perturbation | Method |
|-------------|--------|
| Trailing space injection | Append ` ` to description |
| Zero-width character injection | Insert `\u200b` in tool name |
| Case swap | `swapcase()` on tool name |
| Homoglyph substitution | Replace ASCII chars with look-alikes |
| Synonym substitution in description | Replace key verbs with near-synonyms |
| Parameter reordering | Shuffle schema fields |
| Type widening | Change `str` to `Any` in schema |
| Extra parameter injection | Add undeclared parameter to schema |
| Description truncation | Cut description at word boundary |
| Whitespace normalization attack | Collapse double spaces |

### 7.2 Defense-trilemma K^n budget monitor

`src/research_pipeline/security/trilemma.py` implements a budget monitor based
on the Defense Trilemma formalism (paper 2604.06436).

The trilemma: content-security defense requires trade-offs among:
1. Input sanitization
2. Output filtering
3. Depth (pipeline stage count) limits

**Lipschitz proxy**: The composition of `n` stages has worst-case distortion
proxy `K^n`. The monitor estimates per-stage K from three cheap proxies:
- Token-budget inflation (growth ratio)
- Character edit ratio (Levenshtein / max_len)
- Sanitization delta (characters removed / total)

If the cumulative `K^n` exceeds the configured budget, the pipeline logs a
warning and (optionally) halts.

```python
from research_pipeline.security.trilemma import TrilemmaMonitor, StageDistortion

monitor = TrilemmaMonitor(budget=10.0)
monitor.record(StageDistortion(stage="convert", token_growth=1.2, edit_ratio=0.05,
                               sanitization_delta=0.01, lipschitz_proxy=1.15))
if monitor.is_over_budget():
    raise SecurityError("K^n budget exceeded")
```

---

## 8. Dependency Security

### 8.1 Vulnerability scanning

All dependencies are scanned with `pip-audit` on every CI run:

```bash
uv run pip-audit --skip-editable --ignore-vuln CVE-2026-3219
```

### 8.2 License compatibility

Dependencies are checked for GPL/AGPL licenses that are incompatible with
the MIT license of this project:

```bash
uv run pip-licenses --fail-on="GPL-3.0-only;GPL-3.0-or-later;GPL-2.0-only;GPL-2.0-or-later;AGPL-3.0-only;AGPL-3.0-or-later"
```

> **Note**: The `pymupdf4llm` extra depends on PyMuPDF (AGPL-licensed). This
> extra must **not** be included in CI license checks or production deployments
> unless AGPL obligations are accepted. Use the `docling` or `marker` extras
> instead for license-clean deployments.

---

## 9. Rate Limiting and Abuse Prevention

### 9.1 Per-source rate limiting

Every external API call goes through a `RateLimiter` instance:

- `ArxivRateLimiter`: 3-second hard floor per request (arXiv terms of service)
- `SemanticScholarRateLimiter`: configurable (default: 100 ms)
- `OpenAlexRateLimiter`: configurable (default: 50 ms)

Bypassing the arXiv rate limiter violates arXiv's terms of service and must
never be done.

### 9.2 Retry with backoff

All HTTP calls use the `@retry` decorator (`infra/retry.py`):
- Exponential backoff with jitter
- Respects `Retry-After` response headers
- Maximum retry count configurable per source

---

## 10. Threat Model

| Threat | Likelihood | Mitigation |
|--------|-----------|------------|
| Prompt injection via paper abstract | Medium | Taint tracking; content gates; output filtering |
| ToolTweak: adversarial MCP tool schema modification | Low | SHA-256 hash-pinning via `mcp_guard.py` |
| Secret leakage via logs | Medium | `sanitize_text()`; `detect-secrets` in CI |
| Dependency CVE | Medium | `pip-audit` on every CI run |
| Copyleft license contamination | Low | `pip-licenses` check in CI |
| Excessive API usage / cost | Medium | Per-source rate limiters; configurable quotas |
| Unauthorized file write | Low | HC2 allowlist; agent instructions |
| Destructive command execution | Low | HC3 human-approval requirement |

---

## 11. Security Contacts and Disclosure

For security vulnerabilities, open a private issue at:
https://github.com/grammy-jiang/research-pipeline/issues

Do not disclose vulnerabilities publicly before a fix is available.
