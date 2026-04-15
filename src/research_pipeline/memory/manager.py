"""Memory manager: coordinates all three memory tiers.

Provides a unified interface for the pipeline to interact with memory:

- Stage transitions (working → episodic consolidation)
- Cross-run knowledge (semantic queries)
- Run completion (episodic recording)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from research_pipeline.memory.episodic import Episode, EpisodicMemory
from research_pipeline.memory.semantic import SemanticMemory
from research_pipeline.memory.working import MemoryItem, WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinates three-tier memory: working, episodic, semantic."""

    def __init__(
        self,
        working_capacity: int = 50,
        episodic_path: Path | None = None,
        kg_path: Path | None = None,
    ) -> None:
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory(db_path=episodic_path)
        self.semantic = SemanticMemory(kg_path=kg_path)
        logger.debug("MemoryManager initialized (working=%d)", working_capacity)

    def transition_stage(self, new_stage: str) -> list[MemoryItem]:
        """Handle stage boundary: reset working memory.

        Returns cleared items for potential episodic consolidation.
        """
        cleared = self.working.reset(new_stage)
        logger.info(
            "Memory stage transition → '%s' (cleared %d working items)",
            new_stage,
            len(cleared),
        )
        return cleared

    def record_run(self, episode: Episode) -> None:
        """Record a completed run in episodic memory."""
        self.episodic.record_episode(episode)
        logger.info(
            "Recorded episode: run=%s topic='%s'",
            episode.run_id,
            episode.topic,
        )

    def prior_knowledge(self, topic: str) -> dict[str, Any]:
        """Check what the system already knows about a topic.

        Returns combined info from episodic + semantic memory.
        """
        past_runs = self.episodic.search_by_topic(topic, limit=5)
        topic_overlap = self.semantic.topic_overlap(topic)

        return {
            "past_runs": len(past_runs),
            "past_run_ids": [r.run_id for r in past_runs],
            "past_topics": [r.topic for r in past_runs],
            "known_concepts": topic_overlap.get("concepts", 0),
            "known_methods": topic_overlap.get("methods", 0),
            "known_papers": topic_overlap.get("papers", 0),
        }

    def summary(self) -> dict[str, Any]:
        """Summary of all memory tiers."""
        return {
            "working": self.working.summary(),
            "episodic": {"total_episodes": self.episodic.count()},
            "semantic": self.semantic.stats(),
        }

    def close(self) -> None:
        """Clean up all resources."""
        self.episodic.close()
        self.semantic.close()
