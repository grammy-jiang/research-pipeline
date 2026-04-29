"""Canonical TopicMemoryStore import surface for Phase B B02.

This module exists so Phase B ownership can move to a dedicated store path
without breaking existing callers that still import from
`research_pipeline.briefing.topic_memory`.
"""

from research_pipeline.briefing.topic_memory import TopicMemoryStore

__all__ = ["TopicMemoryStore"]
