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

    def consolidate(self) -> int:
        """Promote episodic memories to semantic store when at capacity.

        Implements the ``consolidate()`` lifecycle operation from the
        memory integration plan: compresses episodes, promotes recurring
        patterns to semantic rules, and prunes stale entries.

        Returns the number of episodes consolidated.
        """
        try:
            from research_pipeline.pipeline.consolidation import (
                EpisodeStore,
                consolidate,
            )

            # Bridge: use a shared episodic DB path so consolidation can
            # read the same episodes that EpisodicMemory recorded.
            store = EpisodeStore(db_path=self.episodic.db_path)
            result = consolidate(store)
            count: int = result.episodes_before - result.episodes_after
            if count:
                logger.info(
                    "Memory consolidation: consolidated %d episode(s) "
                    "(%d rules created, %d rules updated, %d pruned)",
                    count,
                    result.rules_created,
                    result.rules_updated,
                    result.entries_pruned,
                )
            return max(0, count)
        except Exception as exc:
            logger.warning("Memory consolidation skipped: %s", exc)
            return 0

    def between_stages(
        self,
        new_stage: str,
        *,
        consolidation_threshold: int = 20,
    ) -> list[MemoryItem]:
        """Lifecycle hook called at every pipeline stage boundary.

        Implements the ``between_stages()`` hook from the memory integration
        plan (§4.1): resets working memory for the new stage and triggers
        episodic→semantic consolidation when the episodic store exceeds
        ``consolidation_threshold``.

        Args:
            new_stage: Name of the stage that is about to begin.
            consolidation_threshold: Episodic episode count above which
                ``consolidate()`` is called automatically.

        Returns:
            List of working-memory items that were cleared.
        """
        cleared = self.transition_stage(new_stage)
        try:
            ep_count = self.episodic.count()
            if ep_count >= consolidation_threshold:
                logger.debug(
                    "between_stages: episodic count %d ≥ threshold %d → consolidating",
                    ep_count,
                    consolidation_threshold,
                )
                self.consolidate()
        except Exception as exc:
            logger.debug("between_stages consolidation check skipped: %s", exc)
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
