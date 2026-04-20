"""Unit tests for :mod:`research_pipeline.memory.associative`."""

from __future__ import annotations

from research_pipeline.memory.associative import AssociativeLinker


def test_add_creates_link_above_min_weight():
    linker = AssociativeLinker(top_k=3, min_weight=0.1)
    linker.add("a", "transformer attention heads on time series")
    new_links = linker.add("b", "transformer attention on time series forecasting")
    assert new_links, "expected at least one associative link"
    assert new_links[0].source_key == "b"
    assert new_links[0].target_key == "a"
    assert new_links[0].weight >= 0.1


def test_add_below_min_weight_produces_no_link():
    linker = AssociativeLinker(top_k=3, min_weight=0.99)
    linker.add("a", "alpha")
    new_links = linker.add("b", "beta")
    assert new_links == []


def test_top_k_caps_outbound_edges():
    linker = AssociativeLinker(top_k=2, min_weight=0.0)
    linker.add("a", "cat dog bird")
    linker.add("b", "cat dog")
    linker.add("c", "cat")
    new_links = linker.add("d", "cat dog bird fish")
    assert len(new_links) == 2


def test_neighbors_sorted_by_weight_desc():
    linker = AssociativeLinker(top_k=5, min_weight=0.0)
    linker.add("a", "foo bar baz")
    linker.add("b", "foo bar qux")
    linker.add("c", "foo")
    linker.add("d", "foo bar baz qux")
    outs = linker.neighbors("d")
    weights = [lk.weight for lk in outs]
    assert weights == sorted(weights, reverse=True)


def test_traverse_bfs_respects_max_depth():
    linker = AssociativeLinker(top_k=1, min_weight=0.0)
    linker.add("a", "foo")
    linker.add("b", "foo bar")
    linker.add("c", "foo bar baz")
    linker.add("d", "foo bar baz qux")
    reachable_1 = linker.traverse("d", max_depth=1)
    reachable_3 = linker.traverse("d", max_depth=3)
    assert len(reachable_3) >= len(reachable_1)
    assert "d" not in reachable_3


def test_backlinks_returns_inbound_edges():
    linker = AssociativeLinker(top_k=3, min_weight=0.0)
    linker.add("a", "foo bar")
    linker.add("b", "foo bar baz")
    linker.add("c", "foo bar baz qux")
    back_a = linker.backlinks("a")
    assert all(lk.target_key == "a" for lk in back_a)


def test_custom_similarity_function():
    calls = {"n": 0}

    def always_half(a: str, b: str) -> float:
        calls["n"] += 1
        return 0.5

    linker = AssociativeLinker(top_k=2, min_weight=0.3, similarity=always_half)
    linker.add("a", "x")
    linker.add("b", "y")
    linker.add("c", "z")
    assert calls["n"] >= 2
    assert all(lk.weight == 0.5 for lk in linker.neighbors("c"))


def test_bulk_add_returns_total_links():
    linker = AssociativeLinker(top_k=2, min_weight=0.0)
    items = [("a", "foo bar"), ("b", "foo bar baz"), ("c", "foo bar baz qux")]
    total = linker.bulk_add(items)
    assert total == len(linker.links)
    assert len(linker) == 3
