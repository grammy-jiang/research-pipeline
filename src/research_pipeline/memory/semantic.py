"""Semantic memory: cross-run knowledge via knowledge graph.

Wraps the :class:`~research_pipeline.storage.knowledge_graph.KnowledgeGraph`
to provide high-level semantic queries:

- What concepts have been studied before?
- What methods are known for a given problem?
- What claims have been made about a topic?
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticMemory:
    """High-level semantic memory built on the knowledge graph.

    Provides concept-level queries for informing new research runs.
    """

    def __init__(self, kg_path: Path | None = None) -> None:
        # Lazy import to avoid circular deps and optional dep issues
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph

        self._kg = KnowledgeGraph(db_path=kg_path)

    def known_concepts(self, limit: int = 50) -> list[dict[str, str]]:
        """Get concepts the system already knows about."""
        from research_pipeline.storage.knowledge_graph import EntityType

        entities = self._kg.get_entities_by_type(EntityType.CONCEPT)
        result = [{"id": e.entity_id, "name": e.name, **e.properties} for e in entities]
        return result[:limit]

    def known_methods(self, limit: int = 50) -> list[dict[str, str]]:
        """Get methods the system already knows about."""
        from research_pipeline.storage.knowledge_graph import EntityType

        entities = self._kg.get_entities_by_type(EntityType.METHOD)
        result = [{"id": e.entity_id, "name": e.name, **e.properties} for e in entities]
        return result[:limit]

    def related_papers(self, concept_name: str) -> list[dict[str, str]]:
        """Find papers related to a concept."""
        from research_pipeline.storage.knowledge_graph import EntityType

        # Search for the concept entity
        entities = self._kg.get_entities_by_type(EntityType.CONCEPT)
        concept = None
        for e in entities:
            if concept_name.lower() in e.name.lower():
                concept = e
                break
        if concept is None:
            return []

        # Get relations from this concept
        neighbors = self._kg.get_neighbors(concept.entity_id)
        paper_ids: set[str] = set()
        for triple in neighbors:
            if triple.subject_id != concept.entity_id:
                paper_ids.add(triple.subject_id)
            if triple.object_id != concept.entity_id:
                paper_ids.add(triple.object_id)

        papers: list[dict[str, str]] = []
        for pid in paper_ids:
            entity = self._kg.get_entity(pid)
            if entity and entity.entity_type.value == "paper":
                papers.append(
                    {"id": entity.entity_id, "name": entity.name, **entity.properties}
                )
        return papers

    def topic_overlap(self, topic: str) -> dict[str, int]:
        """Check how much the system already knows about a topic.

        Returns counts of known entities related to the topic.
        """
        words = topic.lower().split()
        from research_pipeline.storage.knowledge_graph import EntityType

        overlap: dict[str, int] = {"concepts": 0, "methods": 0, "papers": 0}

        for entity_type, key in [
            (EntityType.CONCEPT, "concepts"),
            (EntityType.METHOD, "methods"),
            (EntityType.PAPER, "papers"),
        ]:
            entities = self._kg.get_entities_by_type(entity_type)
            for e in entities:
                if any(w in e.name.lower() for w in words if len(w) > 3):
                    overlap[key] += 1

        return overlap

    def stats(self) -> dict[str, object]:
        """Get semantic memory statistics."""
        return self._kg.stats()

    def close(self) -> None:
        """Close the underlying knowledge graph connection."""
        self._kg.close()
