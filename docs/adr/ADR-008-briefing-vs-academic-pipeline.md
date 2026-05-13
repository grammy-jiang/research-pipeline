# ADR-008: Separation of Academic Pipeline from Daily Briefing Pipeline

## Status
Accepted

## Date
2024

## Context

Two different workflows emerged in the project:
1. **Deep research** (academic paper pipeline): Given a topic, exhaustively
   search, screen, download, convert, and synthesize academic literature.
   Typical run time: minutes to hours. Output: a long-form synthesis report.

2. **Daily intelligence briefing**: Given a set of configured sources (Hacker
   News, arXiv RSS, newsletters, etc.), poll for the day's items, deduplicate
   and rank them, generate a short briefing, and validate it.
   Typical run time: seconds to 2 minutes. Output: a short daily brief (1000–2000 words).

These two workflows share no significant code path. Mixing them into a single
pipeline would create confusing flag combinations and conflated stage semantics.

## Decision

Implement two **fully separate pipeline subsystems** within the same package:

| Dimension | Academic Pipeline | Briefing Pipeline |
|-----------|-------------------|------------------|
| Entry point | `research-pipeline <stage>` | `research-pipeline brief <stage>` |
| CLI app | Main `app` in `cli/app.py` | `brief_app` sub-app |
| Core stages | 7 (plan→summarize) | 4 (poll→validate) |
| Source code | `src/research_pipeline/` (many modules) | `src/research_pipeline/briefing/` |
| Workspace | `workspace/` | `workspace/briefing/` |
| Databases | `paper_index.db`, `episodic_memory.db`, etc. | `briefing_feedback.db`, `topic_memory.db` |
| MCP tools | Majority (50+) | 14 briefing tools |

Both pipelines share:
- Config system (`config/`)
- Infra layer (`infra/` — rate limiting, retry, caching)
- LLM layer (`llm/`)
- Storage primitives (`storage/`)

## Consequences

**Positive:**
- Each pipeline can evolve independently
- Users who only need briefings don't need to understand the academic pipeline
- Testing is simpler — briefing tests don't need paper fixtures
- The `brief` sub-app can be extended (weekly synthesis, dossier generation)
  without affecting the academic pipeline

**Negative:**
- Some shared concepts (e.g., feedback, rate limiting) are duplicated or
  require careful factoring
- CLI help has two separate command trees — discovery is harder for new users

**Documentation note:** Any document covering the full CLI surface must clearly
distinguish between `research-pipeline <cmd>` (academic pipeline) and
`research-pipeline brief <cmd>` (briefing pipeline).
