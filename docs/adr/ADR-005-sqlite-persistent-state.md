# ADR-005: SQLite for All Persistent State

## Status
Accepted

## Date
2024

## Context

The pipeline needs persistent storage for several independent purposes:
- Cross-run paper deduplication (global paper index)
- Episodic memory of past research runs
- Knowledge graph entities and triples
- Case-based reasoning cases
- Evaluation metrics (Pass@k, Pass[k])
- User feedback for self-improving screening
- Per-run audit logs (3-channel evaluation logging)

Options considered:
1. **Plain files** (JSON/JSONL) — simple but no query support, poor concurrent access
2. **PostgreSQL/MySQL** — full query support but requires external service
3. **SQLite** — embedded, zero-dependency, good query support, sufficient for
   single-node deployment

## Decision

Use **SQLite** for all persistent state. Each database serves a single concern
and is stored at a deterministic path:

| Database | Path | Purpose |
|----------|------|---------|
| `paper_index.db` | `~/.cache/research-pipeline/` | Global paper dedup |
| `episodic_memory.db` | `~/.cache/research-pipeline/` | Past run episodes |
| `knowledge_graph.db` | `~/.cache/research-pipeline/` | Entity+triple KG |
| `cbr_cases.db` | `~/.cache/research-pipeline/` | CBR case store |
| `dual_metrics.db` | `~/.cache/research-pipeline/` | Pass@k / Pass[k] |
| `blinding_audit.db` | `~/.cache/research-pipeline/` | A/B blinding results |
| `audit.db` | `<run_root>/logs/` | Per-run 3-channel eval log |
| `feedback.db` | `<workspace>/` | Paper feedback |
| `briefing_feedback.db` | `<workspace>/` | Briefing feedback |
| `topic_memory.db` | `<workspace>/` | Briefing topic novelty |

## Consequences

**Positive:**
- Zero external dependencies — the pipeline works offline on a laptop
- SQLite supports concurrent reads and single-writer pattern (adequate for
  single-node CLI and MCP server)
- All state can be backed up with a simple file copy

**Negative:**
- SQLite is not suitable for multi-process concurrent writes under high load
- Schema migrations require manual ALTER TABLE or a migration tool
- HC4: Database schema drops must be authored but never executed autonomously
  (hard constraint)

**Note:** Agent-authored schema changes must be reviewed and approved by a human
before execution, per HC4.
