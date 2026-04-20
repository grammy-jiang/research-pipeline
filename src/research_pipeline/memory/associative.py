"""A-MEM associative linking for memory entries.

Based on A-MEM (report Theme 3 / Recommendation 3): every new memory item
triggers a similarity lookup against existing items and the top-k nearest
neighbours are linked to the new item with a weighted associative edge.
Retrieval can then traverse these links to surface semantically related
content even when direct lexical overlap is absent.

The implementation here is deliberately dependency-free: token Jaccard
similarity on whitespace-split lowercased tokens. Callers that want stronger
signals can plug in an embedding-based scorer via the ``similarity`` kwarg.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssociativeLink:
    """Weighted directed link between two memory items."""

    source_key: str
    target_key: str
    weight: float
    reason: str = "token_jaccard"


@dataclass
class _Entry:
    key: str
    tokens: frozenset[str]
    text: str


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t for t in text.lower().split() if len(t) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class AssociativeLinker:
    """Builds and maintains A-MEM-style associative links between items."""

    def __init__(
        self,
        top_k: int = 3,
        min_weight: float = 0.1,
        similarity: Callable[[str, str], float] | None = None,
    ) -> None:
        """Create a linker.

        Args:
            top_k: Maximum number of associative edges to add per new item.
            min_weight: Similarity threshold below which no edge is added.
            similarity: Optional custom similarity function. Defaults to
                token Jaccard on lowercased whitespace tokens.
        """
        self._top_k = max(1, top_k)
        self._min_weight = max(0.0, min(1.0, min_weight))
        self._similarity = similarity
        self._entries: dict[str, _Entry] = {}
        self._links: list[AssociativeLink] = []

    @property
    def links(self) -> list[AssociativeLink]:
        """Return all links in insertion order."""
        return list(self._links)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, key: str, text: str) -> list[AssociativeLink]:
        """Add an item and link it to its top-k most similar predecessors.

        Returns the new links created (may be empty if no predecessor
        clears ``min_weight``).
        """
        if not key:
            raise ValueError("key must be a non-empty string")
        new_entry = _Entry(key=key, tokens=_tokenize(text), text=text)

        scores: list[tuple[str, float]] = []
        for other_key, other in self._entries.items():
            if other_key == key:
                continue
            if self._similarity is not None:
                weight = float(self._similarity(text, other.text))
            else:
                weight = _jaccard(new_entry.tokens, other.tokens)
            if weight >= self._min_weight:
                scores.append((other_key, weight))
        scores.sort(key=lambda s: s[1], reverse=True)
        top = scores[: self._top_k]

        new_links = [
            AssociativeLink(source_key=key, target_key=target, weight=weight)
            for target, weight in top
        ]
        self._entries[key] = new_entry
        self._links.extend(new_links)
        if new_links:
            logger.debug(
                "A-MEM: linked %s → %s (top=%d)",
                key,
                [link.target_key for link in new_links],
                len(new_links),
            )
        return new_links

    def neighbors(self, key: str) -> list[AssociativeLink]:
        """Return all links whose source is *key* sorted by weight desc."""
        out = [lk for lk in self._links if lk.source_key == key]
        out.sort(key=lambda lk: lk.weight, reverse=True)
        return out

    def backlinks(self, key: str) -> list[AssociativeLink]:
        """Return all links whose target is *key* sorted by weight desc."""
        out = [lk for lk in self._links if lk.target_key == key]
        out.sort(key=lambda lk: lk.weight, reverse=True)
        return out

    def traverse(
        self,
        start_key: str,
        max_depth: int = 2,
        min_weight: float | None = None,
    ) -> set[str]:
        """BFS traversal of outbound associative links from *start_key*.

        Args:
            start_key: Starting node.
            max_depth: Max hop count (``1`` = direct neighbours only).
            min_weight: Optional per-hop weight threshold. Defaults to the
                linker's construction-time ``min_weight``.
        """
        threshold = self._min_weight if min_weight is None else min_weight
        visited: set[str] = {start_key}
        frontier: set[str] = {start_key}
        for _ in range(max(0, max_depth)):
            next_frontier: set[str] = set()
            for node in frontier:
                for link in self.neighbors(node):
                    if link.weight >= threshold and link.target_key not in visited:
                        visited.add(link.target_key)
                        next_frontier.add(link.target_key)
            if not next_frontier:
                break
            frontier = next_frontier
        visited.discard(start_key)
        return visited

    def bulk_add(self, items: Iterable[tuple[str, str]]) -> int:
        """Add many items; return total link count across the batch."""
        total = 0
        for key, text in items:
            total += len(self.add(key, text))
        return total
