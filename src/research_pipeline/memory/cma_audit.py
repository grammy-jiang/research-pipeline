"""CMA (Continuum Memory Architecture) six-property completeness audit.

The Continuum Memory Architecture paper (Paper 46 in the research report,
2601.09913) specifies six behavioural properties that a memory system must
satisfy to rise above pure RAG:

1. **Persistence** — state survives across sessions / runs.
2. **Selective retention** — not everything is kept; capacity is bounded.
3. **Retrieval-driven mutation** — memory can be updated by retrieval-based
   operations (non-destructive revision).
4. **Associative routing** — retrieval can follow links between related
   items rather than relying purely on lexical overlap.
5. **Temporal continuity** — old state remains queryable alongside new state
   (versioning / validity intervals).
6. **Consolidation** — episodic experiences can be promoted to structured
   long-term knowledge.

This module provides :class:`CMACompletenessAuditor` which inspects a
:class:`research_pipeline.memory.manager.MemoryManager` instance at runtime
and emits a pass/fail verdict per property together with pointers to the
module(s) satisfying (or failing) each one. Missing any one property "remains
a form of RAG" — the auditor reports this clearly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from research_pipeline.memory.manager import MemoryManager


class CMAProperty(StrEnum):
    """The six CMA completeness properties."""

    PERSISTENCE = "persistence"
    SELECTIVE_RETENTION = "selective_retention"
    RETRIEVAL_DRIVEN_MUTATION = "retrieval_driven_mutation"
    ASSOCIATIVE_ROUTING = "associative_routing"
    TEMPORAL_CONTINUITY = "temporal_continuity"
    CONSOLIDATION = "consolidation"


@dataclass(frozen=True)
class PropertyResult:
    """Audit result for a single CMA property."""

    property: CMAProperty
    passed: bool
    evidence: str
    module: str = ""


@dataclass
class CMAAuditReport:
    """Full CMA audit report for a :class:`MemoryManager`."""

    results: list[PropertyResult] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    is_rag_only: bool = False
    summary: str = ""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> bool:
        """Overall pass only if *every* property holds (CMA paper rule)."""
        return self.failed_count == 0 and self.total == len(CMAProperty)

    def get(self, prop: CMAProperty) -> PropertyResult | None:
        """Return the result for a specific property."""
        for r in self.results:
            if r.property is prop:
                return r
        return None


class CMACompletenessAuditor:
    """Audits a :class:`MemoryManager` against the six CMA properties."""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # ── Individual property checks ────────────────────────────────

    def _check_persistence(self) -> PropertyResult:
        """Persistence: episodic + semantic tiers have a backing store."""
        m = self._manager
        has_episodic_path = getattr(m.episodic, "db_path", None) is not None
        has_semantic_path = getattr(m.semantic, "kg_path", None) is not None
        passed = has_episodic_path or has_semantic_path
        evidence = (
            f"episodic.db_path set={has_episodic_path}; "
            f"semantic.kg_path set={has_semantic_path}"
        )
        return PropertyResult(
            property=CMAProperty.PERSISTENCE,
            passed=passed,
            evidence=evidence,
            module="memory.episodic / memory.semantic",
        )

    def _check_selective_retention(self) -> PropertyResult:
        """Selective retention: working memory has a bounded capacity."""
        working = self._manager.working
        capacity = getattr(working, "capacity", 0)
        passed = capacity > 0 and capacity < 10_000
        return PropertyResult(
            property=CMAProperty.SELECTIVE_RETENTION,
            passed=passed,
            evidence=f"working.capacity={capacity}",
            module="memory.working",
        )

    def _check_retrieval_driven_mutation(self) -> PropertyResult:
        """Retrieval-driven mutation: versioned / non-destructive updates."""
        semantic = self._manager.semantic
        kg = getattr(semantic, "kg", None)
        has_versioned_api = any(
            hasattr(kg, m) for m in ("supersede", "update_with_version", "add_version")
        )
        # Fallback: versioned module is available in the package
        try:
            from research_pipeline.memory import versioned  # noqa: F401

            has_versioned_module = True
        except ImportError:  # pragma: no cover
            has_versioned_module = False
        passed = has_versioned_api or has_versioned_module
        return PropertyResult(
            property=CMAProperty.RETRIEVAL_DRIVEN_MUTATION,
            passed=passed,
            evidence=(
                f"kg has versioned API={has_versioned_api}; "
                f"memory.versioned available={has_versioned_module}"
            ),
            module="memory.versioned / storage.knowledge_graph",
        )

    def _check_associative_routing(self) -> PropertyResult:
        """Associative routing: KG edges or associative links exist."""
        try:
            from research_pipeline.memory import associative  # noqa: F401

            has_associative = True
        except ImportError:
            has_associative = False
        semantic = self._manager.semantic
        kg = getattr(semantic, "kg", None)
        has_edge_api = kg is not None and any(
            hasattr(kg, m) for m in ("add_edge", "neighbors", "related")
        )
        passed = has_associative or has_edge_api
        return PropertyResult(
            property=CMAProperty.ASSOCIATIVE_ROUTING,
            passed=passed,
            evidence=(
                f"memory.associative available={has_associative}; "
                f"kg edge API={has_edge_api}"
            ),
            module="memory.associative / storage.knowledge_graph",
        )

    def _check_temporal_continuity(self) -> PropertyResult:
        """Temporal continuity: episodic history is queryable over time."""
        episodic = self._manager.episodic
        has_history_api = hasattr(episodic, "count") or hasattr(
            episodic, "search_by_topic"
        )
        return PropertyResult(
            property=CMAProperty.TEMPORAL_CONTINUITY,
            passed=has_history_api,
            evidence=f"episodic history API available={has_history_api}",
            module="memory.episodic",
        )

    def _check_consolidation(self) -> PropertyResult:
        """Consolidation: episodic → semantic promotion pathway exists."""
        try:
            from research_pipeline.pipeline import consolidation  # noqa: F401

            has_consolidation = True
        except ImportError:
            has_consolidation = False
        return PropertyResult(
            property=CMAProperty.CONSOLIDATION,
            passed=has_consolidation,
            evidence=f"pipeline.consolidation importable={has_consolidation}",
            module="pipeline.consolidation",
        )

    # ── Public API ────────────────────────────────────────────────

    def audit(self) -> CMAAuditReport:
        """Run all six property checks and build the report."""
        results = [
            self._check_persistence(),
            self._check_selective_retention(),
            self._check_retrieval_driven_mutation(),
            self._check_associative_routing(),
            self._check_temporal_continuity(),
            self._check_consolidation(),
        ]
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        missing = [r.property.value for r in results if not r.passed]
        is_rag_only = failed_count > 0
        summary_parts = [
            f"CMA audit: {passed_count}/{len(results)} properties passed.",
        ]
        if missing:
            summary_parts.append(
                "Missing: " + ", ".join(missing) + " (remains a form of RAG).",
            )
        else:
            summary_parts.append("Full CMA compliance.")
        summary = " ".join(summary_parts)
        report = CMAAuditReport(
            results=results,
            passed_count=passed_count,
            failed_count=failed_count,
            is_rag_only=is_rag_only,
            summary=summary,
        )
        logger.info("CMA audit complete: %s", summary)
        return report
