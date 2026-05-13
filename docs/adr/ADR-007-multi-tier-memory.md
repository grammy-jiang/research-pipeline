# ADR-007: Multi-Tier Memory Architecture

## Status
Accepted

## Date
2024

## Context

A research pipeline that runs repeatedly on related topics needs to:
- Avoid re-downloading papers already seen in prior runs
- Learn from past research strategies that worked well
- Surface connections between concepts across separate runs
- Store episodic history without unbounded memory growth

A single flat list of seen papers is insufficient. The problem maps naturally
to cognitive science's memory model: working memory (short-term, bounded),
episodic memory (autobiographical, chronological), and semantic memory (abstract
knowledge, facts, relations).

## Decision

Implement a **three-tier memory architecture** inspired by neurocognitive models
and the SEA/MLMF framework:

### Tier 1: Working Memory (`memory/working.py`)
- Bounded deque (configurable capacity)
- Holds the current pipeline run's active paper set
- Discarded when run completes (promoted to episodic)

### Tier 2: Episodic Memory (`memory/manager.py` + `episodic_memory.db`)
- SQLite-backed log of completed pipeline runs
- Indexed by topic and date
- Enables `memory-search` and `memory-episodes` commands

### Tier 3: Knowledge Graph (`memory/manager.py` + `knowledge_graph.db`)
- Entity + triple store
- Populated by `kg-ingest` after claim decomposition
- Enables cross-run knowledge queries via `kg-query`

**Supplementary components:**
- **CBR (Case-Based Reasoning)** (`cbr_cases.db`): stores successful past
  strategies for reuse via `cbr-lookup` / `cbr-retain`
- **CMA Audit** (`memory/cma_audit.py`): six-property completeness check for
  memory entries
- **A-MEM Associative Linking** (`memory/associative.py`): Jaccard + BFS to
  find latent connections between past runs
- **MemGPT-style paging** (`memory/paging.py`): fault counters for cold vs.
  warm memory access patterns

**Consolidation** (`memory/manager.py` tool: `consolidation`):
Episodes are periodically compressed into semantic rules (promote if seen in
≥2 runs); stale entries are pruned.

## Consequences

**Positive:**
- Deduplication is cross-run — the same paper is never downloaded twice
- Research quality improves over time as CBR learns effective strategies
- Knowledge graph enables structured queries that flat logs cannot support

**Negative:**
- Six separate databases add operational complexity (backup, migration)
- Memory consolidation must be run manually (no automatic daemon)
- KG quality degrades if `kg-ingest` is not run after each synthesis

**HC4 applies**: Schema drops in any memory database require human approval.
